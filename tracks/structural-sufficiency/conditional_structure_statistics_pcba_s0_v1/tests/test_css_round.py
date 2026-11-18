"""Targeted correctness checks for round css-pcba-s0-v1 (protocol section 6).

Fast tests need no dataset: they build the model from a synthetic normalisation
buffer and use hand-built toy batches.  The data-dependent tests are marked
``slow`` (they load the frozen PCBA feature cache) and are the same checks the
``preflight`` stage runs on the GPU host.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import css_common as C
import css_preflight as PF
import css_model as M

FAKE_NORM = {
    "global_mean": np.zeros(42, dtype=np.float32),
    "global_scale": np.ones(42, dtype=np.float32),
    "scalar_mean": np.zeros(9, dtype=np.float32),
    "scalar_scale": np.ones(9, dtype=np.float32),
}


# ---------------------------------------------------------------------------
# identity / frozen constants
# ---------------------------------------------------------------------------


def test_identity_constants() -> None:
    assert C.ROUND_ID == "css-pcba-s0-v1"
    assert C.PROTOCOL_ID == C.ROUND_ID
    assert C.STUDY_ID == C.ROUND_ID
    assert C.SEED == 0
    assert C.DATASET == "ogbg-molpcba"
    assert (C.N_TRAIN, C.N_VALID, C.N_TEST) == (350_343, 43_793, 43_793)
    assert C.TOTAL_STEPS_40 == 109_520
    assert C.EXPECTED_PARAMETERS == {"E": 831_088, "L": 831_088, "S": 829_040}
    assert sum(C.PARAMETER_BREAKDOWN.values()) == 831_088
    assert C.PARAMETER_BREAKDOWN["query"] == 2_736
    assert C.DROPOUT_GENERATOR_SEED == 230_000
    assert C.CANONICAL_INIT_SEED == 0
    assert C.S_GATE["d40_early_min"] == 0.002
    assert C.S_GATE["e_best_floor"] == 0.262273
    assert C.S_GATE["e_best_high_min"] == 0.268273
    assert C.SUPPORT_MIN_POSITIVE == 20
    assert C.ARMS["E"]["task_query"] and C.ARMS["L"]["task_query"]
    assert not C.ARMS["S"]["task_query"]
    assert C.ARMS["E"]["assigned_pool"] == "res-gpu0"
    assert C.ARMS["L"]["assigned_pool"] == "res-gpu1"


def test_frozen_recipe_and_schedule() -> None:
    recipe = C.RECIPE
    assert recipe["batch_size"] == 128
    assert recipe["epochs"] == 40
    assert recipe["weight_decay"] == 1e-3
    assert recipe["gradient_clip"] == 5.0
    assert recipe["adam_betas"] == (0.9, 0.999)
    assert recipe["warmup_start_lr"] == 3e-5
    assert recipe["learning_rate"] == 3e-4
    assert recipe["end_lr"] == 1e-5
    # the inherited warmup applies (step + 1) / 2738, so step 0 is just above 3e-5
    assert C.scheduled_lr(0, phase=3, kind="all") == pytest.approx(3e-5, abs=1e-7)
    assert C.scheduled_lr(C.STEPS_PER_EPOCH - 1, phase=3, kind="all") == pytest.approx(3e-4)
    assert C.scheduled_lr(C.TOTAL_STEPS_40, phase=3, kind="all") == pytest.approx(1e-5)
    with pytest.raises(KeyError):
        C.scheduled_lr(0, phase=3, kind="base")


def test_frozen_plan_hashes_are_the_historical_ones() -> None:
    assert C.PLAN["global_plan_sha256"] == (
        "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"
    )
    assert C.PLAN["consumed40_prefix_sha256"] == (
        "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58"
    )
    assert C.PLAN["steps_per_epoch"] == 2738


# ---------------------------------------------------------------------------
# model: parameters, initial state, algebra
# ---------------------------------------------------------------------------


def test_parameter_counts_and_initial_state() -> None:
    audit = M.audit_initial_states(FAKE_NORM)
    assert audit["passed"], audit["failures"]
    for arm in ("E", "L"):
        assert audit["parameter_audit"][arm]["parameters"] == 831_088
        assert audit["parameter_audit"][arm]["breakdown"] == C.PARAMETER_BREAKDOWN
    assert audit["parameter_audit"]["S"]["parameters"] == 829_040
    assert audit["S_missing_keys"] == ["qt"]
    assert audit["E_L_initial_state_tensor_identical"]
    assert audit["S_public_state_equal_to_E"]
    assert audit["query_and_output_bias_zero_at_init"] == {
        "q0": 0.0,
        "qt": 0.0,
        "Wc": 0.0,
        "out_bias": 0.0,
    }


def test_initial_states_are_copied_not_coincidental() -> None:
    """Rebuilding in a different order must not change the initial state."""
    a, _ = M.build_model("E", FAKE_NORM)
    torch.manual_seed(999)
    _ = torch.randn(1000)
    b, _ = M.build_model("L", FAKE_NORM)
    for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert na == nb
        assert torch.equal(pa, pb), na
    s, _ = M.build_model("S", FAKE_NORM)
    for name, value in s.state_dict().items():
        assert torch.equal(value, a.state_dict()[name]), name


def _toy_batch():
    return PF._toy_batch(
        n_atoms=3,
        bond_index=[[0, 1], [0, 2]],
        s1=[[0, 1, 0], [0, 2, 1], [1, 0, 0], [2, 0, 1]],
    )


def test_zero_query_makes_arms_identical() -> None:
    batch = _toy_batch()
    outs = {}
    for arm in C.ARM_ORDER:
        model, _ = M.build_model(arm, FAKE_NORM)
        model.eval()
        with torch.no_grad():
            outs[arm] = model(batch)
    assert torch.equal(outs["E"], outs["L"])
    assert torch.equal(outs["E"], outs["S"])


def test_non_exchange_witness() -> None:
    out = PF.gate_non_exchange_witness(FAKE_NORM)
    assert out["passed"], out
    assert out["E_value"] == pytest.approx(math.tanh(1.0) / math.sqrt(2.0), abs=1e-6)
    assert abs(out["L_value"]) < 1e-7


def test_degenerate_equalities() -> None:
    out = PF.gate_degenerate_equalities(FAKE_NORM)
    assert out["passed"], out


def test_task_chunk_equivalence_on_a_toy_batch() -> None:
    batch = _toy_batch()
    ref = None
    for chunk in (128, 16, 8, 4):
        model, _ = M.build_model("E", FAKE_NORM, task_chunk=chunk)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        if ref is None:
            ref = logits
        else:
            assert torch.allclose(logits, ref, atol=1e-6)


def test_dropout_mask_is_per_batch_and_shared_across_chunks() -> None:
    gen = torch.Generator().manual_seed(C.DROPOUT_GENERATOR_SEED)
    mask = PF.T.dropout_mask(64, 128, 64, generator=gen, device=torch.device("cpu"), p=0.10)
    assert mask.shape == (64, 128, 64)
    values = torch.unique(mask).tolist()
    assert np.allclose(sorted(values), [0.0, 1.0 / 0.9], atol=1e-9)
    # a second draw differs (the generator advances once per Train batch)
    gen2 = torch.Generator().manual_seed(C.DROPOUT_GENERATOR_SEED)
    mask2 = PF.T.dropout_mask(64, 128, 64, generator=gen2, device=torch.device("cpu"), p=0.10)
    assert torch.equal(mask, mask2)
    mask3 = PF.T.dropout_mask(64, 128, 64, generator=gen2, device=torch.device("cpu"), p=0.10)
    assert not torch.equal(mask, mask3)


# ---------------------------------------------------------------------------
# S purchase rule
# ---------------------------------------------------------------------------


def test_s_gate_rule_matches_the_protocol() -> None:
    # both triggers fire, floor passes, budget fine
    out = A_s_gate(e_best=0.270, e40=0.270, l40=0.265, used=4.0, projected=3.0)
    assert out["purchased"] and out["label"] == "PURCHASED"
    # d40 exactly at the threshold counts as passing
    out = A_s_gate(e_best=0.263, e40=0.267, l40=0.265, used=4.0, projected=3.0)
    assert out["purchased"]
    # d40 just below the threshold, e_best above the high trigger
    out = A_s_gate(e_best=0.2683, e40=0.2669, l40=0.265, used=4.0, projected=3.0)
    assert out["purchased"]
    # neither trigger fires -> not purchased, reasons reported
    out = A_s_gate(e_best=0.264, e40=0.2655, l40=0.265, used=4.0, projected=3.0)
    assert not out["purchased"]
    assert out["label"] == "NOT_PURCHASED"
    assert out["reasons"]
    # floor failure blocks even when the trigger fires
    out = A_s_gate(e_best=0.260, e40=0.270, l40=0.260, used=4.0, projected=3.0)
    assert not out["purchased"]
    # budget failure blocks
    out = A_s_gate(e_best=0.270, e40=0.270, l40=0.265, used=15.0, projected=3.0)
    assert not out["purchased"]
    assert any("budget" in r for r in out["reasons"])


def A_s_gate(*, e_best, e40, l40, used, projected):
    import css_analyze as A

    return A.s_gate(
        e_best_valid_ap=e_best,
        e_epoch40_valid_ap=e40,
        l_epoch40_valid_ap=l40,
        used_gpu_hours=used,
        projected_s_gpu_hours=projected,
    )


# ---------------------------------------------------------------------------
# data-dependent checks (slow: load the frozen PCache)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_frozen_task_identities(dev_task) -> None:
    assert dev_task.train_rows.size == C.N_TRAIN
    assert dev_task.valid_rows.size == C.N_VALID
    assert dev_task.test_rows.size == C.N_TEST
    assert dev_task.task.spec.hash() == C.FEATURE_SPEC_HASH
    assert dev_task.task.norm["norm_fingerprint"] == C.NORM_FINGERPRINT
    assert dev_task.task.split_partition_sha256 == C.SPLIT_PARTITION_SHA256
    assert dev_task.task.label_artifact_sha256 == C.LABEL_ARTIFACT_SHA256
    assert not np.isfinite(dev_task.task.labels[dev_task.test_rows]).any()
    assert int(np.isfinite(dev_task.labels_dev[dev_task.test_rows]).sum()) > 0


@pytest.mark.slow
def test_order_plan_is_the_frozen_one(plan40) -> None:
    assert len(plan40) == 40
    assert all(int(p["n_batches"]) == 2738 for p in plan40)
    assert plan40[0]["global_plan_sha256"] == C.PLAN["global_plan_sha256"]


@pytest.mark.slow
def test_structure_objects_and_closure(dev_task, plan40) -> None:
    out = PF.gate_structures(dev_task, plan40)
    assert out["passed"], out["failures"]
    # independent closed-form S2 count on one molecule
    import cpc_paths as P

    rows = [int(dev_task.train_rows[0])]
    batch = PF.D.collate_batch(dev_task, rows)
    bi = batch["bond_index"].numpy()
    expected = PF.D.s2_count_closed_form(int(batch["n_atoms"]), bi)
    got = batch["s2_counts"].numpy()
    assert np.array_equal(expected, got)
    assert int(batch["n_s1"]) == 2 * int(batch["n_bonds"])
    assert int(batch["s2_closure"].sum()) <= int(batch["n_s2"])


@pytest.mark.slow
def test_test_labels_cannot_reach_training(dev_task) -> None:
    out = PF.gate_label_isolation(dev_task)
    assert out["passed"], out
    with pytest.raises(RuntimeError):
        dev_task.training_targets(dev_task.test_rows[:1])


@pytest.mark.slow
def test_permutation_and_renumbering_invariance(dev_task, plan40) -> None:
    out = PF.gate_invariances(dev_task, plan40)
    assert out["passed"], out


@pytest.mark.slow
def test_task_chunk_equivalence_on_real_batch(dev_task, plan40) -> None:
    out = PF.gate_task_chunk(dev_task, plan40)
    assert out["passed"], out


@pytest.mark.slow
def test_first_steps_gradient_reachability(dev_task, plan40) -> None:
    import css_train as T

    rows = [int(r) for r in plan40[0]["batches"][0][:8]]
    batch = PF._batch(dev_task, rows, torch.device("cpu"))
    y = torch.as_tensor(dev_task.training_targets(rows), dtype=torch.float32)
    model, _ = M.build_model("E", dev_task.task.norm)
    opt, _ = T.build_optimizer(model)
    gen = T.make_dropout_generator(torch.device("cpu"))
    first = T.train_step(model, opt, batch, y, global_step=0, clip=5.0, generator=gen)
    assert first["module_grad_norms"]["query"] > 0.0
    assert first["module_grad_norms"]["keys"] == 0.0
    second = T.train_step(model, opt, batch, y, global_step=1, clip=5.0, generator=gen)
    assert second["module_grad_norms"]["keys"] > 0.0
    assert second["module_grad_norms"]["values"] > 0.0
    assert second["module_grad_norms"]["task_output"] > 0.0
    assert second["module_grad_norms"]["root_static"] > 0.0
    assert second["module_grad_norms"]["s2_event"] > 0.0


@pytest.mark.slow
def test_prefetch_matches_the_synchronous_pipeline(dev_task, plan40) -> None:
    """Building batches one step ahead must not change a single loss value."""
    import css_train as T

    batches = plan40[0]["batches"][:4]
    losses = {}
    for depth in (0, T.PREFETCH_DEPTH):
        model, _ = M.build_model("E", dev_task.task.norm)
        opt, _ = T.build_optimizer(model)
        gen = T.make_dropout_generator(torch.device("cpu"))
        seen = []
        for i, (_rows, batch, y, _t, _w) in enumerate(T._prefetched(dev_task, batches, depth=depth)):
            out = T.train_step(
                model,
                opt,
                batch,
                torch.as_tensor(y, dtype=torch.float32),
                global_step=i,
                clip=5.0,
                generator=gen,
            )
            seen.append(out["loss"])
        losses[depth] = np.asarray(seen)
    assert np.array_equal(losses[0], losses[T.PREFETCH_DEPTH])
    assert losses[0].size == len(batches)


@pytest.mark.slow
def test_s2_volume_formula_matches_collation(dev_task, plan40) -> None:
    import css_preflight as PF

    per_molecule = D_s2(dev_task)
    assert per_molecule.size == C.N_TRAIN + C.N_VALID + C.N_TEST
    for rows in PF.heaviest_batches(dev_task, plan40, n=2):
        batch = PF.D.collate_batch(dev_task, rows)
        assert int(batch["n_s2"]) == int(per_molecule[np.asarray(rows, dtype=np.int64)].sum())
    batch = PF.D.collate_batch(dev_task, dev_task.train_rows[:16])
    assert int(batch["n_s2"]) == int(per_molecule[dev_task.train_rows[:16]].sum())


def D_s2(dev_task):
    import css_data as D

    return D.s2_counts_per_molecule(dev_task)
