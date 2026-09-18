"""Correctness tests for the MolHIV shared vocabulary-free B-Bag patch encoder.

Fast unit tests only: no official test data, no training, no network.

Coverage (numbered as in the experiment brief, section 15):

A. node permutation invariance
B. edge-order invariance
C. connectivity-free witness (same primitives, different adjacency -> same token)
D. attribute sensitivity (one atom field / one bond field moves the token)
E. multi-field correctness over the OGB categorical schema (no out-of-range misuse)
F. no exact typed vocabulary / no dataset-dependent identity table
G. batch invariance (single molecule vs the same molecule inside a batch)
H. gradient viability (atom / bond embeddings, node MLP, bond MLP, fusion)

Plus: the downstream relational stack is bit-identical to the frozen MolHIV
recurrent pair--centre model when the B-Bag token is replaced by the old typed
lookup, and the custom collate offsets patch groupings across a batch.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc
from tracks.ksvd.experiments.luyin16 import (
    molhiv_shared_bag_patch_encoder as msb,
)

ATOM_DIMS = msb.ATOM_FEATURE_DIMS
BOND_DIMS = msb.BOND_FEATURE_DIMS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class Bag:
    """Primitive container with no edge endpoints (like the real Data path)."""

    def __init__(self, atom_fields, root, dist, patch, bond_fields, edge_patch):
        self.struct_atom_fields = atom_fields
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_bond_fields = bond_fields
        self.struct_edge_patch = edge_patch


def _encoder(seed: int = 0) -> msb.MolhivSharedBagPatchEncoder:
    torch.manual_seed(seed)
    return msb.MolhivSharedBagPatchEncoder().eval()


def _bag_from_patch_nodelists(patches, seed: int = 0):
    """Build a bag container from a list of patch node-counts.

    Each patch gets its root at its first node, distances 0/1/1/2..., one bond
    per non-root node, and all-zero categorical fields except random atom 0
    fields so that different nodes are distinguishable.
    """
    generator = torch.Generator().manual_seed(seed)
    atom_fields = []
    root = []
    dist = []
    patch_ids = []
    bond_fields = []
    edge_patch = []
    for p, n_nodes in enumerate(patches):
        for local in range(n_nodes):
            row = torch.zeros(len(ATOM_DIMS), dtype=torch.long)
            row[0] = int(torch.randint(0, ATOM_DIMS[0], (1,), generator=generator))
            atom_fields.append(row)
            root.append(1 if local == 0 else 0)
            dist.append(0 if local == 0 else (1 if local <= 2 else 2))
            patch_ids.append(p)
        for _ in range(max(n_nodes - 1, 0)):
            bond_fields.append(
                torch.tensor(
                    [
                        int(torch.randint(0, BOND_DIMS[0], (1,), generator=generator)),
                        0,
                        0,
                    ],
                    dtype=torch.long,
                )
            )
            edge_patch.append(p)
    return Bag(
        torch.stack(atom_fields, dim=0),
        torch.tensor(root, dtype=torch.long),
        torch.tensor(dist, dtype=torch.long),
        torch.tensor(patch_ids, dtype=torch.long),
        torch.stack(bond_fields, dim=0)
        if bond_fields
        else torch.zeros((0, len(BOND_DIMS)), dtype=torch.long),
        torch.tensor(edge_patch, dtype=torch.long),
    )


def _molhiv_data(n_patches: int, seed: int = 0) -> Data:
    """A minimal single-molecule PyG ``Data`` with every field the model reads."""
    node_counts = [3] * n_patches
    bag = _bag_from_patch_nodelists(node_counts, seed=seed)
    pairs = [(i, j) for i in range(n_patches) for j in range(i + 1, n_patches)]
    pair_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.zeros((2, 0), dtype=torch.long)
    )
    buckets = torch.tensor(
        [min(i % (mpp.DISTANCE_BUCKETS - 1), mpp.DISTANCE_BUCKETS - 1) for i in range(len(pairs))],
        dtype=torch.long,
    )
    return Data(
        patch_cont=torch.zeros(n_patches, int(mpp.SHELL_WIDTH)),
        parent_token=torch.zeros(n_patches, dtype=torch.long),
        typed_token=torch.zeros(n_patches, dtype=torch.long),
        struct_atom_fields=bag.struct_atom_fields,
        struct_root=bag.struct_root,
        struct_dist=bag.struct_dist,
        struct_patch=bag.struct_patch,
        struct_bond_fields=bag.struct_bond_fields,
        struct_edge_patch=bag.struct_edge_patch,
        pair_index=pair_index,
        pair_relation=torch.zeros(len(pairs), int(mpp.RELATION_WIDTH)),
        pair_bucket=buckets,
        global_context=torch.zeros(1, int(mpp.GLOBAL_WIDTH)),
        y=torch.tensor([float(seed % 2)]),
        num_nodes=n_patches,
    )


# ---------------------------------------------------------------------------
# E: schema-sized embeddings
# ---------------------------------------------------------------------------


def test_field_embedding_sizes_match_ogb_schema():
    encoder = _encoder()
    assert len(encoder.atom_embeddings) == len(ATOM_DIMS) == 9
    assert len(encoder.bond_embeddings) == len(BOND_DIMS) == 3
    for emb, width in zip(encoder.atom_embeddings, ATOM_DIMS):
        assert int(emb.num_embeddings) == int(width)
        assert int(emb.embedding_dim) == 48
    for emb, width in zip(encoder.bond_embeddings, BOND_DIMS):
        assert int(emb.num_embeddings) == int(width)
        assert int(emb.embedding_dim) == 24
    assert int(encoder.root_embedding.num_embeddings) == 2
    assert int(encoder.distance_embedding.num_embeddings) == msb.PATCH_RADIUS + 1
    # normalisation is the 1/sqrt(K) mean over fields
    assert math.isclose(math.sqrt(encoder.n_atom_fields), math.sqrt(9))
    assert math.isclose(math.sqrt(encoder.n_bond_fields), math.sqrt(3))


def test_out_of_range_field_index_raises():
    encoder = _encoder()
    bag = _bag_from_patch_nodelists([3])
    bag.struct_atom_fields = bag.struct_atom_fields.clone()
    bag.struct_atom_fields[0, 0] = int(ATOM_DIMS[0])  # one past the schema
    with pytest.raises((IndexError, RuntimeError)):
        encoder(bag)


# ---------------------------------------------------------------------------
# A: node permutation invariance
# ---------------------------------------------------------------------------


def test_node_relabel_is_invariant():
    encoder = _encoder(1)
    bag = _bag_from_patch_nodelists([3, 2, 4], seed=3)
    with torch.no_grad():
        reference = encoder(bag)
    permutation = torch.tensor([6, 0, 4, 2, 8, 1, 3, 7, 5])
    permuted = Bag(
        bag.struct_atom_fields[permutation],
        bag.struct_root[permutation],
        bag.struct_dist[permutation],
        bag.struct_patch[permutation],
        bag.struct_bond_fields,  # bond grouping does not reference node order
        bag.struct_edge_patch,
    )
    with torch.no_grad():
        shuffled = encoder(permuted)
    assert float((reference - shuffled).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# B: edge-order invariance
# ---------------------------------------------------------------------------


def test_edge_order_is_irrelevant():
    encoder = _encoder(2)
    bag = _bag_from_patch_nodelists([4, 3], seed=5)
    with torch.no_grad():
        reference = encoder(bag)
    order = torch.tensor([3, 0, 4, 1, 2])
    reordered = Bag(
        bag.struct_atom_fields,
        bag.struct_root,
        bag.struct_dist,
        bag.struct_patch,
        bag.struct_bond_fields[order],
        bag.struct_edge_patch[order],
    )
    with torch.no_grad():
        shuffled = encoder(reordered)
    assert float((reference - shuffled).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# C: connectivity-free witness (the definitional B-Bag test)
# ---------------------------------------------------------------------------


def test_connectivity_is_ignored():
    """Identical atom/root/distance/bond multisets, different adjacency.

    Two rooted trees on 5 nodes, root 0, distances ``[0,1,1,2,2]``:
      A: 0-1, 0-2, 1-3, 2-4
      B: 0-1, 0-2, 2-3, 2-4
    The B-Bag containers carry no endpoints at all, so the tokens are identical.
    The adjacency difference is expressed only through (unused) endpoints, which
    the encoder must not read.
    """
    encoder = _encoder(4)
    atom_fields = torch.zeros((5, len(ATOM_DIMS)), dtype=torch.long)
    root = torch.tensor([1, 0, 0, 0, 0])
    dist = torch.tensor([0, 1, 1, 2, 2])
    patch = torch.zeros(5, dtype=torch.long)
    bond_fields = torch.zeros((4, len(BOND_DIMS)), dtype=torch.long)
    edge_patch = torch.zeros(4, dtype=torch.long)

    def with_endpoints(edges):
        bag = Bag(atom_fields.clone(), root.clone(), dist.clone(), patch.clone(),
                  bond_fields.clone(), edge_patch.clone())
        both = edges + [(v, u) for u, v in edges]
        # endpoints exist on the object but must never be consulted
        bag.struct_src = torch.tensor([u for u, _ in both], dtype=torch.long)
        bag.struct_dst = torch.tensor([v for _, v in both], dtype=torch.long)
        return bag

    bag_a = with_endpoints([(0, 1), (0, 2), (1, 3), (2, 4)])
    bag_b = with_endpoints([(0, 1), (0, 2), (2, 3), (2, 4)])
    assert bag_a.struct_src.tolist() != bag_b.struct_src.tolist()
    with torch.no_grad():
        token_a = encoder(bag_a)
        token_b = encoder(bag_b)
        # removing the endpoints entirely changes nothing either
        token_c = encoder(Bag(atom_fields, root, dist, patch, bond_fields, edge_patch))
    assert float((token_a - token_b).abs().max()) == 0.0
    assert float((token_a - token_c).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# D + E: attribute sensitivity over every categorical field
# ---------------------------------------------------------------------------


def test_every_atom_field_moves_the_token():
    encoder = _encoder(5)
    bag = _bag_from_patch_nodelists([3, 3], seed=7)
    with torch.no_grad():
        reference = encoder(bag)
    for field in range(len(ATOM_DIMS)):
        changed = bag.struct_atom_fields.clone()
        current = int(changed[0, field].item())
        changed[0, field] = (current + 1) % int(ATOM_DIMS[field])
        with torch.no_grad():
            moved = encoder(Bag(changed, bag.struct_root, bag.struct_dist,
                                bag.struct_patch, bag.struct_bond_fields,
                                bag.struct_edge_patch))
        assert float((reference - moved).abs().max()) > 1.0e-7, field


def test_every_bond_field_moves_the_token():
    encoder = _encoder(6)
    bag = _bag_from_patch_nodelists([3, 3], seed=11)
    with torch.no_grad():
        reference = encoder(bag)
    for field in range(len(BOND_DIMS)):
        changed = bag.struct_bond_fields.clone()
        current = int(changed[0, field].item())
        changed[0, field] = (current + 1) % int(BOND_DIMS[field])
        with torch.no_grad():
            moved = encoder(Bag(bag.struct_atom_fields, bag.struct_root,
                                bag.struct_dist, bag.struct_patch, changed,
                                bag.struct_edge_patch))
        assert float((reference - moved).abs().max()) > 1.0e-7, field


def test_root_position_and_distance_move_the_token():
    encoder = _encoder(7)
    bag = _bag_from_patch_nodelists([4], seed=13)
    with torch.no_grad():
        reference = encoder(bag)
    root = bag.struct_root.clone()
    root[0] = 0
    root[1] = 1
    with torch.no_grad():
        moved_root = encoder(Bag(bag.struct_atom_fields, root, bag.struct_dist,
                                 bag.struct_patch, bag.struct_bond_fields,
                                 bag.struct_edge_patch))
    dist = bag.struct_dist.clone()
    dist[0] = 2
    with torch.no_grad():
        moved_dist = encoder(Bag(bag.struct_atom_fields, bag.struct_root, dist,
                                 bag.struct_patch, bag.struct_bond_fields,
                                 bag.struct_edge_patch))
    assert float((reference - moved_root).abs().max()) > 1.0e-7
    assert float((reference - moved_dist).abs().max()) > 1.0e-7


# ---------------------------------------------------------------------------
# F: no exact typed vocabulary / dataset-dependent identity
# ---------------------------------------------------------------------------


def test_model_has_no_typed_lookup_and_fixed_size_parameters():
    model = msb.build_model(parent_vocabulary_size=103, seed=0)
    assert getattr(model, "typed_embedding", None) is None
    assert [k for k in model.state_dict() if "typed_embedding" in k] == []
    assert all(int(p.numel()) < 100_000 for p in model.parameters())
    breakdown = msb.parameter_breakdown(model)
    assert breakdown["exact_typed_token_embedding"] == 0
    assert breakdown["total"] == msb._n_params(model)
    assert breakdown["total"] <= msb.PARAM_UPPER_BOUND
    # exact identity storage in the new model is the radius-1 parent table only
    assert msb.blocks_exact_identity(breakdown) == breakdown["radius1_parent_embedding"]
    assert breakdown["radius1_parent_embedding"] == 103 * 16


def test_parameter_breakdown_has_all_required_categories():
    model = msb.build_model(parent_vocabulary_size=103, seed=0)
    breakdown = msb.parameter_breakdown(model)
    for key in (
        "molhiv_atom_field_embeddings",
        "root_distance_embeddings",
        "bond_field_embeddings",
        "bag_node_mlp",
        "bag_bond_mlp",
        "bag_fusion",
        "exact_typed_token_embedding",
        "radius1_parent_embedding",
        "patch_encoder",
        "pair_relation_modules",
        "recurrent_center_update",
        "global_encoder",
        "prediction_head",
        "total",
    ):
        assert key in breakdown, key
    assert breakdown["total"] == sum(
        v
        for k, v in breakdown.items()
        if k not in ("bag_encoder_total", "total")
    )
    # the token width is unchanged, so the patch encoder input width is unchanged
    assert model.patch_encoder.layers[0].in_features == int(mpp.SHELL_WIDTH) + 32 + 16


def test_geometry_matches_frozen_rpc():
    model = msb.build_model(parent_vocabulary_size=103, seed=0)
    assert int(model.patch_hidden) == rpc.H_DIM
    assert int(model.pair_hidden) == rpc.Q_DIM
    assert int(model.recurrence_rounds) == rpc.RECURRENCE_ROUNDS == 2
    assert int(model.token_width) == 32
    assert int(model.bag_encoder.output_dim) == 32
    counts = model.module_call_counts(
        msb.bag_collate([_molhiv_data(6, seed=0)])
    )
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}


# ---------------------------------------------------------------------------
# G: batch invariance
# ---------------------------------------------------------------------------


def test_batch_invariance():
    torch.manual_seed(8)
    model = msb.build_model(parent_vocabulary_size=5, seed=0).eval()
    alone = _molhiv_data(6, seed=1)
    other = _molhiv_data(4, seed=2)
    single_batch = next(iter(msb._make_loader([alone], 1, False, 0)))
    multi_batch = next(iter(msb._make_loader([alone, other], 2, False, 0)))
    with torch.no_grad():
        out_single = model(single_batch)
        out_multi = model(multi_batch)[:1]
    assert float((out_single - out_multi).abs().max()) < 1.0e-4


# ---------------------------------------------------------------------------
# H: gradient viability
# ---------------------------------------------------------------------------


def test_gradient_viability():
    torch.manual_seed(9)
    model = msb.build_model(parent_vocabulary_size=5, seed=0)
    batch = next(
        iter(msb._make_loader([_molhiv_data(6, seed=3), _molhiv_data(5, seed=4)], 2, False, 0))
    )
    report = msb._gradient_viability(model, batch)
    assert report["all_healthy"]
    for name in (
        "atom_field_embeddings",
        "bond_field_embeddings",
        "node_mlp",
        "bond_mlp",
        "fusion",
    ):
        assert report[name]["finite"] and report[name]["nonzero"], name


# ---------------------------------------------------------------------------
# collate offsets
# ---------------------------------------------------------------------------


def test_bag_collate_offsets_patch_ids():
    data_a = _molhiv_data(3, seed=0)
    data_b = _molhiv_data(2, seed=1)
    batch = msb.bag_collate([data_a, data_b])
    assert batch.struct_n_patches == 5
    # graph b's patch ids start after graph a's 3 patches
    assert sorted(set(batch.struct_patch.tolist())) == [0, 1, 2, 3, 4]
    assert int(batch.struct_patch.max()) == 4
    assert int(batch.struct_edge_patch.max()) == 4
    # node-level fields are concatenated, not offset
    expected_nodes = int(data_a.struct_atom_fields.shape[0]) + int(
        data_b.struct_atom_fields.shape[0]
    )
    assert int(batch.struct_atom_fields.shape[0]) == expected_nodes


# ---------------------------------------------------------------------------
# downstream equivalence with the frozen RPC model
# ---------------------------------------------------------------------------


def test_downstream_matches_rpc_when_bag_token_is_typed_lookup():
    """Only the patch-token source differs; the rest of the model is identical."""
    torch.manual_seed(10)
    rpc_model = rpc.MolhivRecurrentPairCentreModel(
        5,
        8,
        patch_hidden=rpc.H_DIM,
        pair_hidden=rpc.Q_DIM,
        token_width=rpc.TOKEN_WIDTH,
        dropout=rpc.DROPOUT,
        center_context=True,
        center_context_hidden=rpc.CENTER_CONTEXT_HIDDEN,
        recurrence_rounds=2,
    ).eval()
    bag_model = msb.build_model(parent_vocabulary_size=8, seed=0)
    bag_state = bag_model.state_dict()
    rpc_state = rpc_model.state_dict()
    shared = {
        key: value
        for key, value in rpc_state.items()
        if key in bag_state and bag_state[key].shape == value.shape
    }
    bag_model.load_state_dict(shared, strict=False)

    class TypedTwin(nn.Module):
        def __init__(self, reference: nn.Module) -> None:
            super().__init__()
            self.reference = reference

        def forward(self, data: Data) -> torch.Tensor:
            return self.reference.typed_embedding(data.typed_token)

    bag_model.bag_encoder = TypedTwin(rpc_model)
    bag_model.eval()

    batch = msb.bag_collate([_molhiv_data(6, seed=5), _molhiv_data(4, seed=6)])
    with torch.no_grad():
        out_bag = bag_model(batch)
        out_rpc = rpc_model(batch)
    assert float((out_bag - out_rpc).abs().max()) < 1.0e-6
