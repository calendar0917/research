"""CPU correctness tests for W2-DYNAMIC (the single model of this round).

Every load-bearing claim of ``PROTOCOL.md`` section 3 is exercised here without
CUDA: exact parameter budget, bitwise W2 seed-0 part, zero-residual identity,
invariances, molecule isolation, the two-layer cross-layer dependency, an
independent numpy re-implementation of the aggregation, and the zero-bond graph.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import dy_common as C
import dy_model as M


# ---------------------------------------------------------------------------
# construction / budget
# ---------------------------------------------------------------------------


def test_parameter_budget_is_exact(model, model_audit):
    audit = model_audit["parameters"]
    assert audit["total"] == 1_128_468
    assert audit["w2_total"] == 994_324
    assert audit["new_total"] == 134_144
    assert audit["groups"] == {
        "w2_base": 848_096,
        "w2_readout": 146_228,
        "bond_proj": 1_792,
        "blocks.0": 66_176,
        "blocks.1": 66_176,
    }
    assert audit["all_match"]
    assert int(sum(p.numel() for p in model.parameters())) == 1_128_468
    assert C.EXPECTED_PARAMETERS["total"] == 1_128_468


def test_block_shapes(model):
    for block in model.blocks:
        assert tuple(block.norm.normalized_shape) == (128,)
        assert block.fc1.in_features == 128 and block.fc1.out_features == 256
        assert block.fc2.in_features == 256 and block.fc2.out_features == 128
    assert model.bond_proj.in_features == 13 and model.bond_proj.out_features == 128
    assert len(model.blocks) == 2


def test_w2_part_is_bitwise_the_seed0_constructor(model, model_audit, reference):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ours = C.state_subset(state, new=False)
    ref = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    cmp = M.compare_states(ours, ref)
    assert cmp["all_equal"] and cmp["n_tensors"] == 57
    assert model_audit["w2_trunk_fingerprint_matches_recorded_w2"]
    assert (
        model_audit["w2_trunk_init_fingerprint"]
        == "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d"
    )


def test_new_module_initialisation_is_private_and_zeroed(model, model_audit):
    init = model_audit["init"]
    assert init["global_rng_unchanged"] is True
    assert init["linear_layers_initialised"] == 5
    assert init["layernorm_layers_initialised"] == 2
    assert init["private_generator_seed"] == 17
    for block in model.blocks:
        assert float(block.fc2.weight.abs().max()) == 0.0
        assert float(block.fc2.bias.abs().max()) == 0.0
        assert float(block.fc1.weight.abs().max()) > 0.0
        assert float(block.norm.weight.min()) == 1.0


def test_constructing_dynamic_model_never_advances_global_rng(spec, norm):
    torch.manual_seed(1234)
    M.build_dynamic_model(spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", verify=False)
    after = torch.get_rng_state().clone()
    # only the W2 constructor may draw (it is the historical one); the dynamic
    # modules themselves must not move the public stream further
    torch.manual_seed(1234)
    M.untrained_w2(spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")
    after_w2 = torch.get_rng_state().clone()
    assert torch.equal(after, after_w2)


# ---------------------------------------------------------------------------
# initial identity / invariances
# ---------------------------------------------------------------------------


def _logits(model, batch):
    model.eval()
    with torch.inference_mode():
        return model(batch).detach().float()


def test_zero_residual_reproduces_untrained_w2(model, reference, batch):
    a = _logits(model, batch)
    b = _logits(reference, batch)
    assert float((a - b).abs().max()) == 0.0


def test_atom_renumbering_invariance(task, model, batch):
    perm, p = C.BM.permute_atoms(batch)
    a = _logits(model, batch)
    b = _logits(model, perm)
    assert float((a - b).abs().max()) <= 1e-5
    assert int(p.numel()) == int(batch["atom_feat"].shape[0])


def test_edge_direction_flip_invariance(model, batch):
    flip = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    flip["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
    a = _logits(model, batch)
    b = _logits(model, flip)
    assert float((a - b).abs().max()) <= 1e-5


def test_molecule_isolation(task, model):
    rows = [int(task.valid_rows[0]), int(task.valid_rows[1])]
    batch = C.PC.collate(task.cache, rows)
    edges = M.check_edges_intra_molecule(batch)
    assert edges["checked"] and edges["both_directions"] and edges["each_physical_bond_twice"]
    base = _logits(model, batch)
    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    mask = (batch["atom_batch"] == 1).unsqueeze(1)
    gen = torch.Generator().manual_seed(7)
    noise = torch.randn(perturbed["atom_feat"].shape, generator=gen)
    perturbed["atom_feat"] = torch.where(mask, perturbed["atom_feat"] + noise, perturbed["atom_feat"])
    after = _logits(model, perturbed)
    assert float((base[0] - after[0]).abs().max()) == 0.0
    assert float((base[1] - after[1]).abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# the message path itself
# ---------------------------------------------------------------------------


def test_bond_directions_are_bidirectional_and_intra_molecular(batch):
    idx = batch["bond_index"]
    src = torch.cat([idx[:, 0], idx[:, 1]])
    dst = torch.cat([idx[:, 1], idx[:, 0]])
    assert src.numel() == 2 * idx.shape[0] == dst.numel()
    assert torch.equal(torch.sort(src).values, torch.sort(dst).values)
    same = batch["atom_batch"][src] == batch["atom_batch"][dst]
    assert bool(same.all())


def test_layer1_reads_h0_and_layer2_reads_h1(nondegenerate, batch):
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for i in (0, 1):
        def _hook(_m, inp, _out, key=f"blocks.{i}"):
            captured[key] = inp[0].detach().clone()

        handles.append(nondegenerate.blocks[i].register_forward_hook(_hook))
    try:
        nondegenerate.eval()
        with torch.inference_mode():
            _a, _b, _s, h0 = nondegenerate.encode_h0(batch)
            h1, h2 = nondegenerate.dynamic_layers(h0, batch)
    finally:
        for h in handles:
            h.remove()
    assert float((captured["blocks.0"] - h0).abs().max()) == 0.0
    assert float((captured["blocks.1"] - h1).abs().max()) == 0.0
    assert float((captured["blocks.1"] - h0).abs().max()) > 1e-6


def test_chain_graph_cross_layer_dependency(nondegenerate, spec):
    width = int(C.DYNAMIC["width"])
    gen = torch.Generator().manual_seed(424242)
    h0 = torch.randn(3, width, generator=gen)
    bond_feat = torch.randn(2, int(spec.bond_feat_dim), generator=gen)
    batch = {
        "bond_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "bond_feat": bond_feat,
        "atom_batch": torch.zeros(3, dtype=torch.long),
        "n_graphs": 1,
    }
    nondegenerate.eval()

    def run(h):
        with torch.inference_mode():
            return nondegenerate.dynamic_layers(h, batch)

    h1_a, h2_a = run(h0)
    delta = torch.zeros_like(h0)
    delta[0] = torch.randn(width, generator=gen) * 0.5
    h1_b, h2_b = run(h0 + delta)
    # A -- B -- C: one layer cannot reach C from A, two layers can
    assert float((h1_a[2] - h1_b[2]).abs().max()) == 0.0
    assert float((h1_a[1] - h1_b[1]).abs().max()) > 1e-6
    assert float((h2_a[2] - h2_b[2]).abs().max()) > 1e-6


def test_python_reference_matches_module_aggregation(nondegenerate, spec):
    width = int(C.DYNAMIC["width"])
    gen = torch.Generator().manual_seed(777)
    h = torch.randn(4, width, generator=gen)
    bond_feat = torch.randn(3, int(spec.bond_feat_dim), generator=gen)
    batch = {
        "bond_index": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
        "bond_feat": bond_feat,
        "atom_batch": torch.zeros(4, dtype=torch.long),
        "n_graphs": 1,
    }
    nondegenerate.eval()
    with torch.inference_mode():
        src, dst, edge_msg = nondegenerate.bond_directions(batch)
        for block in nondegenerate.blocks:
            z = block.norm(h)
            msg = torch.zeros_like(z).index_add_(
                0, dst, torch.nn.functional.silu(z[src] + edge_msg)
            )
            delta = block(h, src=src, dst=dst, bond_msg=edge_msg)
            # independent pure-numpy implementation (float64)
            eps = float(block.norm.eps)
            hn = h.numpy().astype(np.float64)
            z_np = (hn - hn.mean(1, keepdims=True)) / np.sqrt(hn.var(1, keepdims=True) + eps)
            z_np = z_np * block.norm.weight.detach().numpy() + block.norm.bias.detach().numpy()
            msg_np = np.zeros_like(z_np)
            em = edge_msg.numpy().astype(np.float64)
            for e in range(src.numel()):
                x = z_np[src[e].item()] + em[e]
                msg_np[dst[e].item()] += x / (1.0 + np.exp(-x))
            a = (z_np + msg_np) @ block.fc1.weight.detach().numpy().T + block.fc1.bias.detach().numpy()
            hidden = a / (1.0 + np.exp(-a))
            delta_np = hidden @ block.fc2.weight.detach().numpy().T + block.fc2.bias.detach().numpy()
            assert float(np.abs(msg.numpy().astype(np.float64) - msg_np).max()) <= 2e-5
            assert float(np.abs(delta.numpy().astype(np.float64) - delta_np).max()) <= 2e-5


def test_zero_bond_graph_message_is_exactly_zero(nondegenerate, spec):
    zi = torch.zeros(0, dtype=torch.long)
    batch = {
        "n_graphs": 1,
        "atom_feat": torch.zeros(1, int(spec.atom_feat_dim)),
        "atom_type": torch.zeros(1, dtype=torch.long),
        "atom_scalar": torch.zeros(1, int(spec.n_scalars)),
        "atom_batch": torch.zeros(1, dtype=torch.long),
        "atom_off": torch.zeros(1, dtype=torch.long),
        "bond_feat": torch.zeros((0, int(spec.bond_feat_dim))),
        "bond_type": zi,
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
        "bond_batch": zi,
        "nbr_atom": zi,
        "nbr_code": zi,
        "path_atom": zi,
        "path_cat": zi,
        "path_end63": zi,
        "path_end": zi,
        "path_closure": zi,
        "path_graph": zi,
        "global": torch.zeros(1, int(spec.global_dim)),
        "atom_counts": torch.ones(1, dtype=torch.long),
        "bond_counts": torch.zeros(1, dtype=torch.long),
    }
    nondegenerate.eval()
    with torch.inference_mode():
        src, dst, edge_msg = nondegenerate.bond_directions(batch)
        assert src.numel() == 0 and dst.numel() == 0
        _a, _b, _s, h0 = nondegenerate.encode_h0(batch)
        z = nondegenerate.blocks[0].norm(h0)
        msg = torch.zeros_like(z)
        if src.numel():
            msg = msg.index_add_(0, dst, torch.nn.functional.silu(z[src] + edge_msg))
        h1, h2 = nondegenerate.dynamic_layers(h0, batch)
        logits = nondegenerate(batch)
    assert float(msg.abs().max()) == 0.0
    assert torch.isfinite(h2).all() and torch.isfinite(logits).all()


def test_usage_metrics_shape(model, batch):
    u = M.usage_metrics(model, batch)
    assert u["R1"] == 0.0 and u["R2"] == 0.0 and u["Rtotal"] == 0.0
    assert u["finite_fraction_h2"] == 1.0
    assert u["finite_logits"] is True
    assert set(u["fc2_weight_norm"]) == {"blocks.0", "blocks.1"}


# ---------------------------------------------------------------------------
# diagnostics used by the post-training interventions
# ---------------------------------------------------------------------------


def test_shuffle_keeps_dst_multiplicity_and_source_multiset(nondegenerate, batch):
    src, dst, edge_msg = nondegenerate.bond_directions(batch, message="real")
    shuffled = M.shuffle_src_within_molecule(src, dst, batch["atom_batch"], seed=0)
    assert torch.equal(torch.sort(shuffled).values, torch.sort(src).values)
    mol = batch["atom_batch"][dst]
    shuffled2 = M.shuffle_src_within_molecule(src, dst, batch["atom_batch"], seed=0)
    assert torch.equal(shuffled, shuffled2)  # deterministic
    same_before = (batch["atom_batch"][src] == mol).all()
    same_after = (batch["atom_batch"][shuffled] == mol).all()
    assert bool(same_before) and bool(same_after)
    assert not torch.equal(shuffled, src)


def test_message_off_gives_zero_message(nondegenerate, batch):
    src, dst, _msg = nondegenerate.bond_directions(batch, message="off")
    assert src.numel() == 0 and dst.numel() == 0
    with torch.inference_mode():
        a = nondegenerate(batch, message="real")
        b = nondegenerate(batch, message="off")
    assert float((a - b).abs().max()) > 0.0


# ---------------------------------------------------------------------------
# inherited protocol identity
# ---------------------------------------------------------------------------


def test_frozen_recipe_and_plan(task):
    assert C.BATCH_SIZE == 128 and C.TOTAL_EPOCHS == 40
    assert C.STEPS_PER_EPOCH == 2738 and C.TOTAL_STEPS_40 == 109_520
    assert float(C.RECIPE["learning_rate"]) == 3e-4
    assert float(C.RECIPE["warmup_start_lr"]) == 3e-5
    assert float(C.RECIPE["end_lr"]) == 1e-5
    assert float(C.RECIPE["weight_decay"]) == 1e-3
    assert float(C.RECIPE["dropout"]) == 0.10
    assert float(C.RECIPE["gradient_clip"]) == 5.0
    assert C.scheduled_lr(0, phase=3, kind="all") == pytest.approx(3e-5 + (3e-4 - 3e-5) / 2738)
    assert C.scheduled_lr(2737, phase=3, kind="all") == pytest.approx(3e-4)
    plan = C.SD.make_plan40(task, seed=C.SEED)
    assert plan[0]["global_plan_sha256"] == C.FROZEN_PLAN["global_plan_sha256"]
    prefix = C.sha256_bytes("|".join(p["row_order_sha256"] for p in plan).encode())
    assert prefix == C.FROZEN_PLAN["consumed40_prefix_sha256"]


def test_split_identity_and_test_isolation(task):
    assert int(task.train_rows.size) == 350_343
    assert int(task.valid_rows.size) == 43_793
    assert int(task.test_rows.size) == 43_793
    assert not np.isfinite(task.labels[task.test_rows]).any()
    assert task.split_partition_sha256 == C.SOURCE["split_partition_sha256"]
    assert task.label_artifact_sha256 == C.SOURCE["label_artifact_sha256"]
    assert task.norm["norm_fingerprint"] == C.SOURCE["norm_fingerprint"]
    assert C.test_rows_sha256_check(task.test_rows) == C.BOC_TEST_ROWS_NPY_SHA256


def test_no_extra_batch_field_is_read(model, reference, batch):
    class Traced(dict):
        def __init__(self, data):
            super().__init__(data)
            self.read = set()

        def __getitem__(self, key):
            self.read.add(str(key))
            return super().__getitem__(key)

    a = Traced(batch)
    b = Traced(batch)
    _logits(reference, a)
    _logits(model, b)
    assert b.read == a.read
    assert not (b.read - a.read)


def test_parameter_groups_share_one_optimizer_group(spec, norm):
    import dy_train as T

    model, _ = M.build_dynamic_model(spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")
    optimizer, audit = T.build_optimizer(model)
    assert len(optimizer.param_groups) == 1
    assert audit["n_parameters"] == 1_128_468
    assert audit["single_group_same_lr_for_w2_and_dynamic"] is True


def test_masked_bce_ignores_missing_labels():
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    logits = torch.zeros(2, 2)
    loss = C.MET.masked_bce_loss(logits, y)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(3), torch.tensor([1.0, 0.0, 1.0])
    )
    assert float(loss) == pytest.approx(float(manual), rel=0, abs=1e-7)
    assert math.isfinite(float(loss))


def test_freeze_seal_matches_the_inherited_gated_reader():
    import cc_data as CD

    manifest = {
        "schema": C.SCHEMA_TEST_FREEZE,
        "logical_models": None,
        "predictions": {"X": {"test_logits_sha256": "abc", "test_logits_shape": [43793, 128]}},
        "test_rows_sha256": "def",
        "selection_manifest_sha256": "ghi",
    }
    assert C.freeze_seal(manifest) == CD.freeze_seal(manifest)
    assert C.SCHEMA_TEST_FREEZE == "lcrc-pcba-test-prediction-freeze-v1"


def test_reference_logits_are_frozen_and_hashed():
    refs = C.verify_frozen_reference_logits()
    assert set(refs) == {
        "W2_best_valid",
        "W2_epoch40",
        "RC_REAL_best_valid",
        "RC_REAL_epoch40",
    }
    for rec in refs.values():
        assert rec["verified"] and rec["shape"] == [43793, 128]
