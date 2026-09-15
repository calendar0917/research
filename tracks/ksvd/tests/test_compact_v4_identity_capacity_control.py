"""Static / unit tests for the ZINC cell-A identity capacity control.

No training is performed and the official test split is never loaded.  These
tests pin the fairness of the A0/A1/A2 comparison:

* exact parameter accounting (A1/A2 total within 1% of A0, adapter sized to
  absorb the released typed-lookup parameters);
* every non-typed tensor remains byte-identical to the frozen cell A;
* the fixed identity code is deterministic, seed-independent, unit-norm and
  distinguishable, and carries no gradient;
* A2 is exactly invariant to a typed-token permutation (no typed identity
  information) while A1 is sensitive;
* the weight-tied T=2 recurrence and finite forward/backward are unchanged.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_identity_capacity_control as icc,
)
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp


def _synthetic_batch() -> Data:
    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
    ):
        n = len(graph.nodes)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        buckets = torch.tensor(
            [min(max(i % 5, 0), 4) for i in range(len(pairs))], dtype=torch.long
        )
        datasets.append(
            Data(
                patch_cont=torch.zeros(n, int(zpp.SHELL_WIDTH)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.arange(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), int(zpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.zeros(1, int(zpp.GLOBAL_WIDTH)),
                topology_features=torch.zeros(1, topo_width),
                y=torch.zeros(1),
                num_nodes=n,
            )
        )
    return next(iter(zpp._make_loader(datasets, 128, False, 0)))


def test_01_exact_parameter_accounting_and_fairness():
    payload = icc.params()
    by_condition = {row["condition"]: row for row in payload["rows"]}
    assert by_condition["A0"]["total_params"] == 85763
    assert by_condition["A0"]["typed_lookup_trainable_params"] == 36420
    assert by_condition["A1"]["total_params"] == 85740
    assert by_condition["A2"]["total_params"] == 85740
    assert by_condition["A1"]["shared_adapter_trainable_params"] == 36397
    assert by_condition["A2"]["shared_adapter_trainable_params"] == 36397
    # A1/A2 match A0 within the pre-registered +/-1% total-parameter budget.
    for condition in ("A1", "A2"):
        assert by_condition[condition]["total_matches_A0_within_1pct"]
        assert (
            abs(by_condition[condition]["total_params"] - 85763) / 85763
            <= icc.UNIFIED_TOLERANCE
        )
    assert payload["a1_a2_total_identical"]
    assert payload["a1_a2_adapter_identical"]
    # Parent lookup untouched in every condition.
    for row in payload["rows"]:
        assert row["parent_lookup_params"] == 256
        assert row["unified_graph_width"] == 334


def test_02_shared_tensors_bit_identical_to_frozen_cell_a():
    _a1, _copied, reference_state = icc.build_condition("A1", 0)
    a1 = icc.build_model("A1", 0)
    a2 = icc.build_model("A2", 0)
    shared = [
        key for key in reference_state if not key.startswith("typed_embedding.")
    ]
    assert len(shared) == 43
    a1_state, a2_state = a1.state_dict(), a2.state_dict()
    for key in shared:
        assert torch.equal(a1_state[key], reference_state[key])
        assert torch.equal(a2_state[key], reference_state[key])
    # adapters identical between A1 and A2
    adapter_keys = [key for key in a1_state if key.startswith("typed_embedding.")]
    assert adapter_keys
    for key in adapter_keys:
        assert torch.equal(a1_state[key], a2_state[key])


def test_03_fixed_code_is_deterministic_seed_independent_and_distinguishable():
    code = icc.fixed_identity_codes(icc.TYPED_VOCABULARY_SIZE)
    assert code.shape == (6785, 16)
    assert not code.requires_grad
    assert torch.equal(code, icc.fixed_identity_codes(icc.TYPED_VOCABULARY_SIZE))
    a1_seed0 = icc.build_model("A1", 0).identity_code
    a1_seed1 = icc.build_model("A1", 1).identity_code
    assert torch.equal(a1_seed0, a1_seed1)  # seed-independent
    assert torch.equal(a1_seed0, code)
    stats = icc._code_statistics(code)
    assert stats["exact_duplicate_rows"] == 0
    assert stats["abs_cosine_max"] < 0.999
    assert abs(stats["norm_min"] - 1.0) < 1e-5
    assert abs(stats["norm_max"] - 1.0) < 1e-5
    assert stats["oov_code_norm"] > 0.0


def test_04_a2_has_no_typed_identity_a1_does():
    batch = _synthetic_batch()
    a1 = icc.build_model("A1", 0).eval()
    a2 = icc.build_model("A2", 0).eval()
    permuted = batch.clone()
    permuted.typed_token = batch.typed_token.flip(0)
    with torch.no_grad():
        out_a1 = a1(batch)
        out_a1_perm = a1(permuted)
        out_a2 = a2(batch)
        out_a2_perm = a2(permuted)
    assert float((out_a2 - out_a2_perm).abs().max()) == 0.0
    assert float((out_a1 - out_a1_perm).abs().max()) > 0.0
    with torch.no_grad():
        slot_a2 = a2.typed_embedding.identity_slot(batch.typed_token)
        slot_a1 = a1.typed_embedding.identity_slot(batch.typed_token)
    assert float(slot_a2.abs().max()) == 0.0
    assert float(slot_a1.abs().max()) > 0.0


def test_05_weight_tying_recurrence_and_finite_backward():
    batch = _synthetic_batch()
    model = icc.build_model("A1", 0)
    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
    assert int(model.recurrence_rounds) == 2
    assert model.recurrence_enabled and model.recurrence_mode == "refresh"
    assert int(model.patch_hidden) == 64
    assert int(model.pair_hidden) == 16

    model.zero_grad(set_to_none=True)
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    assert torch.isfinite(out).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.identity_code.grad is None
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.identity_adapter.parameters()
    )
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parent_embedding.parameters()
    )
