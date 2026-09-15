"""Tests for the shared connectivity-free BAG patch encoder (B-bag).

Fast unit / static tests only: no official test data, no full training.
Requirements covered (numbered as in the experiment brief):

1. node relabeling preserves ``e_bag``
2. changing an atom type changes the representation
3. changing the bond-type multiset changes the representation
4. changing the root designation changes the representation
5. changing the root-distance multiset changes the representation
6. identical primitive multisets + different connectivity => identical ``e_bag``
7. the same adversarial pair moves the B-full (connectivity-aware) encoder
8. the bag forward never reads edge endpoints
9. the candidate has no vocab-sized ``typed_embedding``
10. output width is 16
11. q=16, h=64, T=2
12. total params are budget-matched to B-full (84,495 +/- 1%)
13. forward / backward are finite
14. edge order is irrelevant
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_bag_patch_encoder as sbpe,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
    SharedBagPatchEncoder,
    SharedStructuralPatchEncoder,
)


class _BagBatch:
    def __init__(self, atom, root, dist, patch, edge_patch, bond) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_edge_patch = edge_patch
        self.struct_bond = bond


class _FullBatch(_BagBatch):
    def __init__(self, atom, root, dist, patch, edge_patch, bond, src, dst) -> None:
        super().__init__(atom, root, dist, patch, edge_patch, bond)
        self.struct_src = src
        self.struct_dst = dst


def _path_batch() -> _FullBatch:
    """Typed path 0-1-2-3 with distinct atom and bond types."""
    undirected = [(0, 1), (1, 2), (2, 3)]
    both = undirected + [(v, u) for u, v in undirected]
    src = torch.tensor([u for u, _ in both], dtype=torch.long)
    dst = torch.tensor([v for _, v in both], dtype=torch.long)
    return _FullBatch(
        atom=torch.tensor([0, 1, 2, 3]),
        root=torch.tensor([1, 0, 0, 0]),
        dist=torch.tensor([0, 1, 2, 1]),
        patch=torch.zeros(4, dtype=torch.long),
        edge_patch=torch.zeros(len(both), dtype=torch.long),
        bond=torch.tensor([0, 0, 1, 1, 2, 2]),
        src=src,
        dst=dst,
    )


# ---------------------------------------------------------------------------
# 1-5, 14: invariance / sensitivity
# ---------------------------------------------------------------------------


def test_node_relabel_is_invariant():
    torch.manual_seed(0)
    encoder = SharedBagPatchEncoder().eval()
    batch = _path_batch()
    with torch.no_grad():
        e = encoder(batch)
    perm = torch.tensor([0, 2, 3, 1])
    permuted = _BagBatch(
        batch.struct_atom[perm],
        batch.struct_root[perm],
        batch.struct_dist[perm],
        batch.struct_patch[perm],
        batch.struct_edge_patch,
        batch.struct_bond,
    )
    with torch.no_grad():
        e2 = encoder(permuted)
    assert float((e - e2).abs().max()) < 1.0e-6


def test_edge_order_is_irrelevant():
    torch.manual_seed(1)
    encoder = SharedBagPatchEncoder().eval()
    batch = _path_batch()
    with torch.no_grad():
        e = encoder(batch)
    order = torch.tensor([5, 0, 2, 4, 1, 3])
    shuffled = _BagBatch(
        batch.struct_atom,
        batch.struct_root,
        batch.struct_dist,
        batch.struct_patch,
        batch.struct_edge_patch[order],
        batch.struct_bond[order],
    )
    with torch.no_grad():
        e2 = encoder(shuffled)
    assert float((e - e2).abs().max()) < 1.0e-6


def test_primitive_changes_move_representation():
    torch.manual_seed(2)
    encoder = SharedBagPatchEncoder().eval()
    batch = _path_batch()
    with torch.no_grad():
        e = encoder(batch)
        atom2 = batch.struct_atom.clone()
        atom2[1] = 7
        e_atom = encoder(
            _BagBatch(
                atom2,
                batch.struct_root,
                batch.struct_dist,
                batch.struct_patch,
                batch.struct_edge_patch,
                batch.struct_bond,
            )
        )
        bond2 = batch.struct_bond.clone()
        bond2[0] = 3
        e_bond = encoder(
            _BagBatch(
                batch.struct_atom,
                batch.struct_root,
                batch.struct_dist,
                batch.struct_patch,
                batch.struct_edge_patch,
                bond2,
            )
        )
        root2 = torch.tensor([0, 1, 0, 0])
        e_root = encoder(
            _BagBatch(
                batch.struct_atom,
                root2,
                batch.struct_dist,
                batch.struct_patch,
                batch.struct_edge_patch,
                batch.struct_bond,
            )
        )
        dist2 = batch.struct_dist.clone()
        dist2[1] = 2
        e_dist = encoder(
            _BagBatch(
                batch.struct_atom,
                batch.struct_root,
                dist2,
                batch.struct_patch,
                batch.struct_edge_patch,
                batch.struct_bond,
            )
        )
    assert float((e - e_atom).abs().max()) > 1.0e-6
    assert float((e - e_bond).abs().max()) > 1.0e-6
    assert float((e - e_root).abs().max()) > 1.0e-6
    assert float((e - e_dist).abs().max()) > 1.0e-6


# ---------------------------------------------------------------------------
# 6-7: the integrity gate (connectivity isolation)
# ---------------------------------------------------------------------------


def test_connectivity_is_ignored_by_bag_and_seen_by_bfull():
    torch.manual_seed(3)
    batch_a, batch_b = sbpe._adversarial_connectivity_pair()
    # primitives are identical; only endpoints differ
    assert torch.equal(batch_a.struct_atom, batch_b.struct_atom)
    assert torch.equal(batch_a.struct_root, batch_b.struct_root)
    assert torch.equal(batch_a.struct_dist, batch_b.struct_dist)
    assert sorted(batch_a.struct_bond.tolist()) == sorted(
        batch_b.struct_bond.tolist()
    )
    assert torch.equal(batch_a.struct_edge_patch, batch_b.struct_edge_patch)
    assert batch_a.struct_src.tolist() != batch_b.struct_src.tolist()

    bag = SharedBagPatchEncoder().eval()
    full = SharedStructuralPatchEncoder(output_dim=16).eval()
    with torch.no_grad():
        bag_a = bag(batch_a)
        bag_b = bag(batch_b)
        full_a = full(batch_a)
        full_b = full(batch_b)
    assert float((bag_a - bag_b).abs().max()) == 0.0
    assert float((full_a - full_b).abs().max()) > 1.0e-6


def test_bag_forward_does_not_read_endpoints():
    torch.manual_seed(4)
    encoder = SharedBagPatchEncoder().eval()
    batch = _path_batch()
    bag_only = _BagBatch(
        batch.struct_atom,
        batch.struct_root,
        batch.struct_dist,
        batch.struct_patch,
        batch.struct_edge_patch,
        batch.struct_bond,
    )
    with torch.no_grad():
        e1 = encoder(batch)
        e2 = encoder(bag_only)
    assert float((e1 - e2).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 8-9: collate offsets / no endpoints required
# ---------------------------------------------------------------------------


def test_bag_collate_offsets_patch_and_edge_patch():
    def make(n_nodes: int, n_patches: int, offsets):
        data = Data(num_nodes=n_nodes, y=torch.tensor([0.0]))
        data.patch_cont = torch.zeros((n_nodes, 4))
        data.patch_context = torch.zeros((n_nodes, 0))
        data.typed_token = torch.zeros(n_nodes, dtype=torch.long)
        data.parent_token = torch.zeros(n_nodes, dtype=torch.long)
        data.structural_token = torch.zeros(n_nodes, dtype=torch.long)
        data.structural_coarse = torch.zeros((n_nodes, 4))
        data.global_context = torch.zeros((1, zpp.GLOBAL_WIDTH))
        data.pair_index = torch.zeros((2, 0), dtype=torch.long)
        data.pair_relation = torch.zeros((0, 23))
        data.pair_bucket = torch.zeros(0, dtype=torch.long)
        data.topology_features = torch.zeros((1, 25))
        data.struct_atom = torch.zeros(n_nodes, dtype=torch.long)
        data.struct_root = torch.ones(n_nodes, dtype=torch.long)
        data.struct_dist = torch.zeros(n_nodes, dtype=torch.long)
        data.struct_patch = torch.zeros(n_nodes, dtype=torch.long)
        data.struct_src, data.struct_dst, data.struct_bond, data.struct_edge_patch = (
            offsets
        )
        return data

    data_a = make(
        2,
        2,
        (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
        ),
    )
    data_b = make(
        2,
        2,
        (
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([0]),
        ),
    )
    batch = sbpe.bag_collate([data_a, data_b])
    # graph b patch ids are offset by graph a's 2 patches
    assert batch.struct_patch.tolist() == [0, 0, 2, 2]
    # graph b edge patch id is offset by graph a's 2 patches
    assert batch.struct_edge_patch.tolist() == [0, 1, 2]
    # graph b node block starts at 2
    assert batch.struct_src.tolist() == [0, 1, 2]
    assert batch.struct_dst.tolist() == [1, 0, 3]
    assert batch.struct_n_patches == 4


# ---------------------------------------------------------------------------
# 10-13: model-level checks
# ---------------------------------------------------------------------------


def test_candidate_has_no_typed_lookup_and_is_budget_matched():
    model = sbpe.build_candidate(0)
    assert getattr(model, "typed_embedding", None) is None
    assert [k for k in model.state_dict() if "typed_embedding" in k] == []
    total = sbpe._n_params(model)
    assert sbpe.PARAM_LOWER <= total <= sbpe.PARAM_UPPER
    assert abs(total - sbpe.REFERENCE_BFULL_PARAMS) / sbpe.REFERENCE_BFULL_PARAMS <= (
        sbpe.BUDGET_MATCH_TOLERANCE
    )
    assert total == 84511
    assert sbpe._n_params(model.structural_encoder) == 35168


def test_candidate_preserves_cell_a_geometry_and_width():
    model = sbpe.build_candidate(0)
    assert int(model.pair_hidden) == 16
    assert int(model.patch_hidden) == 64
    assert int(model.recurrence_rounds) == 2
    assert int(model.patch_encoder_hidden) == 64
    assert int(model.global_encoder_hidden) == 32
    assert int(model.structural_encoder.output_dim) == 16
    assert int(model.unified_graph_width) == 334
    # the patch encoder input width is unchanged because e_bag is token_width
    assert model.patch_encoder.layers[0].in_features == 146 + 0 + 16 + 8


def test_model_encode_is_connectivity_invariant():
    """End-to-end: two batches differing only in endpoints encode equally."""
    batch_a, batch_b = sbpe._adversarial_connectivity_pair()
    collated_a = sbpe.bag_collate([_pad_full(batch_a)])
    collated_b = sbpe.bag_collate([_pad_full(batch_b)])
    model = sbpe.build_candidate(0).eval()
    with torch.no_grad():
        out_a = model.encode(collated_a)
        out_b = model.encode(collated_b)
    assert float((out_a - out_b).abs().max()) == 0.0


def _pad_full(batch: _FullBatch) -> Data:
    """Turn an adversarial pair into a single-patch batch for ``model.encode``."""
    batch_size = 1
    n_nodes = int(batch.struct_atom.numel())
    data = Data(num_nodes=batch_size, y=torch.tensor([0.0]))
    data.patch_cont = torch.zeros((batch_size, 146))
    data.patch_context = torch.zeros((batch_size, 0))
    data.typed_token = torch.zeros(batch_size, dtype=torch.long)
    data.parent_token = torch.zeros(batch_size, dtype=torch.long)
    data.structural_token = torch.zeros(batch_size, dtype=torch.long)
    data.structural_coarse = torch.zeros((batch_size, 4))
    data.global_context = torch.zeros((1, zpp.GLOBAL_WIDTH))
    data.pair_index = torch.zeros((2, 0), dtype=torch.long)
    data.pair_relation = torch.zeros((0, 23))
    data.pair_bucket = torch.zeros(0, dtype=torch.long)
    data.topology_features = torch.zeros((1, 25))
    data.struct_atom = batch.struct_atom
    data.struct_root = batch.struct_root
    data.struct_dist = batch.struct_dist
    data.struct_patch = torch.zeros(n_nodes, dtype=torch.long)
    data.struct_edge_patch = torch.zeros(
        batch.struct_bond.numel(), dtype=torch.long
    )
    data.struct_bond = batch.struct_bond
    data.struct_src = batch.struct_src
    data.struct_dst = batch.struct_dst
    return data


def test_recurrent_weight_tying_and_forward_backward():
    model = sbpe.build_candidate(0)
    data = Data(num_nodes=1, y=torch.tensor([0.0]))
    data.patch_cont = torch.zeros((1, 146))
    data.patch_context = torch.zeros((1, 0))
    data.typed_token = torch.zeros(1, dtype=torch.long)
    data.parent_token = torch.zeros(1, dtype=torch.long)
    data.structural_token = torch.zeros(1, dtype=torch.long)
    data.structural_coarse = torch.zeros((1, 4))
    data.global_context = torch.zeros((1, zpp.GLOBAL_WIDTH))
    data.pair_index = torch.tensor([[0], [0]], dtype=torch.long)
    data.pair_relation = torch.zeros((1, 23))
    data.pair_bucket = torch.zeros(1, dtype=torch.long)
    data.topology_features = torch.zeros((1, 25))
    data.struct_atom = torch.tensor([0, 1, 2])
    data.struct_root = torch.tensor([1, 0, 0])
    data.struct_dist = torch.tensor([0, 1, 1])
    data.struct_patch = torch.zeros(3, dtype=torch.long)
    data.struct_edge_patch = torch.zeros(2, dtype=torch.long)
    data.struct_bond = torch.tensor([0, 0])
    data.struct_src = torch.tensor([0, 1])
    data.struct_dst = torch.tensor([1, 0])
    batch = sbpe.bag_collate([data])

    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
    model.train()
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    assert bool(torch.isfinite(out).all())
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parameters()
    )


def test_bfull_reference_candidate_is_unchanged():
    """The B-full candidate must still build to its frozen 84,495 params."""
    model = sspe.build_candidate(0)
    assert sspe._n_params(model) == sbpe.REFERENCE_BFULL_PARAMS
