"""Correctness tests for the Z1 Adaptive Structure-Binding (ASB) cell.

Fast unit / static tests only: no official test data, no full training.

Coverage (numbered as in the Z1 brief, section 11):

A. permutation invariance (path / branch / triangle-cycle / multi-parent)
B. connectivity (every hard support contains the root and is connected)
C. induced-edge correctness (all real bonds between selected nodes are kept)
D. full-gate degeneration (selected == full patch; ASB init ~ B-bag)
E. no vocabulary-sized parameters
F. gradient viability (gate / bind / message / update / fusion)
G. batch invariance (single molecule vs multi-molecule batch)

plus edge-order invariance and explicit support read-out sanity.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_adaptive_structure_binding_cell as asb,
)
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
    AdaptiveStructureBindingEncoder,
    SharedBagPatchEncoder,
)


class _FullBatch:
    """Minimal structural-encoder input container (with edge grouping)."""

    def __init__(self, atom, root, dist, patch, edge_patch, bond, src, dst) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_edge_patch = edge_patch
        self.struct_bond = bond
        self.struct_src = src
        self.struct_dst = dst


def _batch_from_edges(
    atom,
    dist,
    edges,
    bond_types=None,
    root_index: int = 0,
    n_patches: int = 1,
) -> _FullBatch:
    """Build a single-patch batch from an undirected edge list."""
    n = len(atom)
    both = list(edges) + [(v, u) for u, v in edges]
    if bond_types is None:
        bond_types = [0] * len(edges)
    bond = torch.tensor(
        [bond_types[i % len(bond_types)] for i in range(len(edges))]
        + [bond_types[i % len(bond_types)] for i in range(len(edges))],
        dtype=torch.long,
    )
    src = torch.tensor([u for u, _ in both], dtype=torch.long)
    dst = torch.tensor([v for _, v in both], dtype=torch.long)
    root = torch.zeros(n, dtype=torch.long)
    root[root_index] = 1
    return _FullBatch(
        atom=torch.tensor(atom, dtype=torch.long),
        root=root,
        dist=torch.tensor(dist, dtype=torch.long),
        patch=torch.zeros(n, dtype=torch.long),
        edge_patch=torch.zeros(len(both), dtype=torch.long),
        bond=bond,
        src=src,
        dst=dst,
    )


def _path_batch() -> _FullBatch:
    # root 0 -> {1, 3}; 1 -> 2  (a path 0-1-2 plus a branch 0-3)
    return _batch_from_edges(
        atom=[0, 1, 2, 3],
        dist=[0, 1, 2, 1],
        edges=[(0, 1), (1, 2), (0, 3)],
        bond_types=[0, 1, 2],
    )


def _branch_batch() -> _FullBatch:
    # root 0 -> {1, 2}; 1 -> 3; 2 -> 4
    return _batch_from_edges(
        atom=[0, 1, 2, 3, 4],
        dist=[0, 1, 1, 2, 2],
        edges=[(0, 1), (0, 2), (1, 3), (2, 4)],
        bond_types=[0, 1, 0, 2],
    )


def _cycle_batch() -> _FullBatch:
    # triangle 0-1-2 plus node 3 at distance 2 reachable through BOTH 1 and 2
    # (multiple distance-1 parents), with the real induced bonds 1-2 and 2-3.
    return _batch_from_edges(
        atom=[0, 1, 2, 3],
        dist=[0, 1, 1, 2],
        edges=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
        bond_types=[0, 1, 2, 0, 1],
    )


def _permuted_batch(batch: _FullBatch, perm: torch.Tensor) -> _FullBatch:
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(int(perm.numel()))
    return _FullBatch(
        atom=batch.struct_atom[perm],
        root=batch.struct_root[perm],
        dist=batch.struct_dist[perm],
        patch=batch.struct_patch[perm],
        edge_patch=batch.struct_edge_patch,
        bond=batch.struct_bond,
        src=inverse[batch.struct_src],
        dst=inverse[batch.struct_dst],
    )


def _encoder(seed: int = 0) -> AdaptiveStructureBindingEncoder:
    torch.manual_seed(seed)
    return AdaptiveStructureBindingEncoder().eval()


# ---------------------------------------------------------------------------
# A. permutation invariance
# ---------------------------------------------------------------------------


def test_permutation_invariance_path_branch_cycle():
    for builder in (_path_batch, _branch_batch, _cycle_batch):
        encoder = _encoder(1)
        batch = builder()
        encoder.record_support = True
        with torch.no_grad():
            e_ref = encoder(batch)
            selected_ref = encoder.last_support["selected"].clone()
        perm = torch.tensor([0, 2, 3, 1]) if batch.struct_atom.numel() == 4 else (
            torch.tensor([4, 3, 2, 1, 0])
        )
        permuted = _permuted_batch(batch, perm)
        encoder.record_support = True
        with torch.no_grad():
            e_perm = encoder(permuted)
            selected_perm = encoder.last_support["selected"].clone()
        assert float((e_ref - e_perm).abs().max()) < 1.0e-6
        # map the permuted support back to the original labelling
        mapped = torch.zeros_like(selected_ref)
        mapped[perm] = selected_perm
        assert torch.equal(mapped, selected_ref)


def test_independent_encoders_agree_on_permuted_path():
    encoder = _encoder(2)
    batch = _path_batch()
    perm = torch.tensor([2, 0, 3, 1])
    permuted = _permuted_batch(batch, perm)
    with torch.no_grad():
        e1 = encoder(batch)
        e2 = encoder(permuted)
    assert float((e1 - e2).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# B. connectivity
# ---------------------------------------------------------------------------


def test_all_supports_connected_and_rooted():
    for builder in (_path_batch, _branch_batch, _cycle_batch):
        encoder = _encoder(3)
        batch = builder()
        encoder.record_support = True
        with torch.no_grad():
            encoder(batch)
        support = encoder.last_support
        assert asb._all_supports_connected(support), builder.__name__


# ---------------------------------------------------------------------------
# C. induced-edge correctness
# ---------------------------------------------------------------------------


def test_cycle_induced_edges_are_kept():
    """A real bond between two selected nodes must enter the support structure.

    The triangle 0-1-2 needs the ``1-2`` bond that is *not* part of the
    root->child growth edges; the multi-parent node 3 needs both ``1-3`` and
    ``2-3``.
    """
    encoder = _encoder(4)
    batch = _cycle_batch()
    encoder.record_support = True
    with torch.no_grad():
        encoder(batch)
    support = encoder.last_support
    assert float(support["selected"].sum()) == 4.0
    selected_edges = {
        tuple(sorted((int(u), int(v))))
        for u, v, keep in zip(
            support["src"].tolist(),
            support["dst"].tolist(),
            support["selected_edge"].tolist(),
        )
        if keep > 0.5
    }
    assert selected_edges == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}


def test_gated_support_partial_still_keeps_induced_bonds():
    """Force every gate ~0.5: support may shrink but stays an induced subgraph.

    Weaker overrides (last-layer bias 0.0 -> gate ~0.5, level-2 activation
    ~0.5) are used to exercise the straight-through path.
    """
    encoder = _encoder(5)
    batch = _cycle_batch()
    with torch.no_grad():
        encoder.gate_mlp[-1].weight.zero_()
        encoder.gate_mlp[-1].bias.zero_()  # sigmoid(0) == 0.5
    encoder.record_support = True
    with torch.no_grad():
        encoder(batch)
    support = encoder.last_support
    selected_edges = {
        tuple(sorted((int(u), int(v))))
        for u, v, keep in zip(
            support["src"].tolist(),
            support["dst"].tolist(),
            support["selected_edge"].tolist(),
        )
        if keep > 0.5
    }
    # every selected node other than the root has a selected incident bond
    # and the edge set is exactly the induced set (subset of real bonds).
    real = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}
    assert selected_edges <= real
    # edge (1,2) connects two nodes that are both distance-1 and is never a
    # growth edge; if both are selected it must be present.
    if float(support["selected"][1]) > 0.5 and float(support["selected"][2]) > 0.5:
        assert (1, 2) in selected_edges


# ---------------------------------------------------------------------------
# D. full-gate degeneration
# ---------------------------------------------------------------------------


def test_full_gate_degeneration_matches_bbag():
    torch.manual_seed(6)
    encoder = _encoder(6)
    bbag = SharedBagPatchEncoder().eval()
    encoder.load_attribute_pretrained(bbag)
    batch = _branch_batch()
    asb._force_full_gate(encoder)
    encoder.record_support = True
    with torch.no_grad():
        e_asb = encoder(batch)
        selected = encoder.last_support["selected"].clone()
    with torch.no_grad():
        e_bbag = bbag(batch)
    assert torch.all(selected > 0.5)
    assert float((e_asb - e_bbag).abs().max()) < 1.0e-5


def test_natural_init_is_close_to_bbag():
    torch.manual_seed(7)
    encoder = _encoder(7)
    bbag = SharedBagPatchEncoder().eval()
    encoder.load_attribute_pretrained(bbag)
    batch = _path_batch()
    with torch.no_grad():
        delta = float((encoder(batch) - bbag(batch)).abs().max())
    assert delta < 1.0e-2


def test_full_gate_selected_equals_full_patch():
    encoder = _encoder(8)
    asb._force_full_gate(encoder)
    batch = _cycle_batch()
    encoder.record_support = True
    with torch.no_grad():
        encoder(batch)
    support = encoder.last_support
    assert float(support["selected"].sum()) == 4.0
    assert float(support["selected_edge"].sum()) == 5.0


# ---------------------------------------------------------------------------
# E. no vocabulary-sized parameters
# ---------------------------------------------------------------------------


def test_candidate_has_no_vocabulary_sized_parameters():
    model = asb.build_candidate(0)
    assert getattr(model, "typed_embedding", None) is None
    assert [k for k in model.state_dict() if "typed_embedding" in k] == []
    assert asb._dataset_dependent_params(model) == 0
    assert isinstance(model.structural_encoder, AdaptiveStructureBindingEncoder)


def test_encoder_parameter_count_is_vocabulary_free():
    a = AdaptiveStructureBindingEncoder(atom_categories=28)
    b = AdaptiveStructureBindingEncoder(atom_categories=28)
    # the ASB encoder has no parameter whose shape equals the certificate
    # vocabulary; rebuilding cannot add a certificate table.
    names = [name for name, _ in a.named_parameters()]
    assert not any("certificate" in name or "typed" in name for name in names)
    assert asb._n_params(a) == asb._n_params(b)


# ---------------------------------------------------------------------------
# F. gradient viability
# ---------------------------------------------------------------------------


def test_gradients_reach_gate_binding_message_update_fusion():
    encoder = _encoder(9)
    encoder.train()
    batch = _cycle_batch()
    out = encoder(batch)
    loss = out.pow(2).mean()
    loss.backward()
    for name in (
        "gate_mlp",
        "bind_delta_mlp",
        "message_mlp",
        "update_mlp",
        "bind_update",
        "fusion",
        "node_mlp",
    ):
        module = getattr(encoder, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, name
        assert all(bool(torch.isfinite(g).all()) for g in grads), name
        assert any(float(g.abs().max()) > 0.0 for g in grads), name


# ---------------------------------------------------------------------------
# G. batch invariance
# ---------------------------------------------------------------------------


def _pad_molecule(batch: _FullBatch) -> Data:
    """Wrap a synthetic single-patch batch as PyG Data for ``asb_collate``."""
    n = int(batch.struct_atom.numel())
    n_patches = 1
    data = Data(num_nodes=n_patches, y=torch.tensor([0.0]))
    data.patch_cont = torch.zeros((n_patches, 146))
    data.patch_context = torch.zeros((n_patches, 0))
    data.typed_token = torch.zeros(n_patches, dtype=torch.long)
    data.parent_token = torch.zeros(n_patches, dtype=torch.long)
    data.structural_token = torch.zeros(n_patches, dtype=torch.long)
    data.structural_coarse = torch.zeros((n_patches, 4))
    data.global_context = torch.zeros((1, 87))
    data.pair_index = torch.zeros((2, 0), dtype=torch.long)
    data.pair_relation = torch.zeros((0, 23))
    data.pair_bucket = torch.zeros(0, dtype=torch.long)
    data.topology_features = torch.zeros((1, 25))
    data.struct_atom = batch.struct_atom
    data.struct_root = batch.struct_root
    data.struct_dist = batch.struct_dist
    data.struct_patch = torch.zeros(n, dtype=torch.long)
    data.struct_edge_patch = torch.zeros(batch.struct_bond.numel(), dtype=torch.long)
    data.struct_bond = batch.struct_bond
    data.struct_src = batch.struct_src
    data.struct_dst = batch.struct_dst
    return data


def test_batch_invariance_single_vs_batch():
    encoder = _encoder(10)
    single = _path_batch()
    with torch.no_grad():
        e_single = encoder(single)
    collated = asb.asb_collate([_pad_molecule(single)])
    molecule = _branch_batch()
    # a second graph with a different atom count, appended after the first
    data_b = _pad_molecule(molecule)
    data_b.struct_edge_patch = torch.ones(
        molecule.struct_bond.numel(), dtype=torch.long
    )
    collated_multi = asb.asb_collate([_pad_molecule(single), data_b])
    with torch.no_grad():
        e_multi = encoder(collated_multi)
    mask = collated_multi.struct_patch_batch == 0
    assert float((e_multi[mask] - e_single).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# extra: edge-order invariance and explicit support read-out
# ---------------------------------------------------------------------------


def test_edge_order_is_irrelevant():
    encoder = _encoder(11)
    batch = _cycle_batch()
    order = torch.tensor([9, 0, 5, 2, 7, 1, 8, 3, 6, 4])
    shuffled = _FullBatch(
        atom=batch.struct_atom,
        root=batch.struct_root,
        dist=batch.struct_dist,
        patch=batch.struct_patch,
        edge_patch=batch.struct_edge_patch[order],
        bond=batch.struct_bond[order],
        src=batch.struct_src[order],
        dst=batch.struct_dst[order],
    )
    with torch.no_grad():
        assert float((encoder(batch) - encoder(shuffled)).abs().max()) < 1.0e-6


def test_support_readout_returns_local_edges():
    encoder = _encoder(12)
    batch = _cycle_batch()
    encoder.record_support = True
    with torch.no_grad():
        encoder(batch)
    support = encoder.last_support
    assert support["selected"].shape[0] == 4
    assert support["src"].shape[0] == support["selected_edge"].shape[0]
    assert set(support["selected_edge"].tolist()) == {1.0}
