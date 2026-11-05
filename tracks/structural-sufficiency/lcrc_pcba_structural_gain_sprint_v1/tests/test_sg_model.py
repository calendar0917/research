"""CPU correctness tests for the structural-gain sprint arms.

These tests do not touch a GPU; they cover the checkpoint/optimizer audit,
architecture and parameter accounting, zero-residual identity, adjacency and
context semantics, gradient flow, the frozen LR schedule, the masked BCE and
Train/Valid/Test isolation.  GPU epoch-0 AP reproduction is in
``code/sg_preflight.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import sg_common as C
import sg_checkpoint as CK
import sg_data as D
import sg_model as M
import pc_metrics as MET
import pc_model as BM


def _arm(task, base_init, source, arm):
    model, audit = M.build_arm(
        arm,
        task.spec,
        task.norm,
        C.N_TASKS,
        seed=C.SEED,
        device="cpu",
        base_init_state=base_init,
        warm_start_state=source["state_dict"],
        zero_residual=True,
    )
    return model, audit


def _batch(task, rows=(0, 1, 2, 3)):
    return BM.collate(task.cache, list(rows))


def test_source_checkpoint_and_optimizer_restore(task, source, reference):
    ref, info, base_init = reference
    audit = CK.verify_source_metadata(source, ref)
    assert audit["state_dict_tensors"] == 37
    assert audit["state_dict_keys_shapes_match_frozen_c1"] is True
    assert audit["adamw_step_values"] == [24642]
    assert all(audit["metadata_checks"].values())
    assert len(audit["state_dict_fingerprint"]) == 64

    model, _ = _arm(task, base_init, source, "M")
    optimizer, opt_audit = M.make_optimizer_warm(model, source["optimizer_state_dict"])
    assert opt_audit["group0_identical_to_source"] is True
    assert opt_audit["source_step_restored"] == [24642.0]
    assert opt_audit["moments_reinitialised"] is False
    assert opt_audit["n_param_groups_after"] == 2
    assert opt_audit["new_param_group_size"] > 0
    assert opt_audit["new_param_state_empty_at_start"] is True
    # The source moments are still present for all 31 base parameters.
    state = M.optimizer_state_cpu(optimizer)
    assert set(range(31)).issubset(set(state["state"]))
    assert all(
        float(state["state"][pid]["step"].item()) == 24642.0 for pid in range(31)
    )
    assert all(int(pid) not in state["state"] for pid in range(31, 31 + opt_audit["new_param_group_size"]))


def test_architecture_and_parameter_counts(task, source, reference):
    _, _, base_init = reference
    counts = {}
    audits = {}
    for arm in C.ARM_ORDER:
        model, audit = _arm(task, base_init, source, arm)
        counts[arm] = int(sum(p.numel() for p in model.parameters()))
        audits[arm] = audit
        assert audit["base_parameter_count"] == 31
        assert audit["total_parameter_count"] == counts[arm]
        for block in model.blocks or []:
            assert torch.count_nonzero(block.fc2.weight).item() == 0
            assert torch.count_nonzero(block.fc2.bias).item() == 0
    assert counts["C"] == 245_184
    assert counts["F"] - counts["C"] == 33_408
    assert counts["M"] - counts["C"] == 34_304
    assert counts["V"] - counts["C"] == 42_624
    assert abs(counts["M"] - counts["F"]) / counts["F"] <= 0.05
    assert audits["C"]["new_parameter_count"] == 0
    # F and M share the same FFN initial state.
    f_model, _ = _arm(task, base_init, source, "F")
    m_model, _ = _arm(task, base_init, source, "M")
    for i, (fb, mb) in enumerate(zip(f_model.blocks, m_model.blocks)):
        assert torch.equal(fb.norm.weight, mb.norm.weight)
        assert torch.equal(fb.fc1.weight, mb.fc1.weight)
        assert torch.equal(fb.fc1.bias, mb.fc1.bias)
        assert torch.equal(fb.fc2.weight, mb.fc2.weight)
        assert torch.equal(fb.fc2.bias, mb.fc2.bias)
    # Only F/M/V add blocks; C keeps the original model footprint.
    assert audits["C"]["block_mode"] == "none"
    assert audits["F"]["block_mode"] == "ffn"
    assert audits["M"]["block_mode"] == "mp"
    assert audits["V"]["block_mode"] == "mp_context"


def test_global_rng_untouched_by_private_init(task, source, reference):
    """Adding new modules must not consume the public dropout RNG stream."""
    _, _, base_init = reference
    states = {}
    for arm in ("C", "V"):
        torch.manual_seed(777)
        torch.get_rng_state()
        M.build_arm(
            arm,
            task.spec,
            task.norm,
            C.N_TASKS,
            seed=C.SEED,
            device="cpu",
            base_init_state=base_init,
            warm_start_state=source["state_dict"],
            zero_residual=True,
        )
        states[arm] = torch.get_rng_state().clone()
    assert torch.equal(states["C"], states["V"])


def test_zero_residual_identity_eval(task, source, reference):
    ref, _, base_init = reference
    ref.load_state_dict(source["state_dict"], strict=True)
    ref.eval()
    batch = _batch(task)
    with torch.no_grad():
        ref_logits = ref(batch)
    for arm in C.ARM_ORDER:
        model, _ = _arm(task, base_init, source, arm)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        diff = float((ref_logits - logits).abs().max().item())
        assert diff <= 1e-6, f"arm {arm}: zero-residual identity diff {diff}"
        assert logits.shape == (4, C.N_TASKS)
        assert torch.isfinite(logits).all()


def test_zero_residual_identity_train_dropout(task, source, reference):
    ref, _, base_init = reference
    ref.load_state_dict(source["state_dict"], strict=True)
    batch = _batch(task)
    ref.train()
    torch.manual_seed(123)
    ref_logits = ref(batch).detach()
    for arm in ("F", "M", "V"):
        model, _ = _arm(task, base_init, source, arm)
        model.train()
        torch.manual_seed(123)
        logits = model(batch).detach()
        diff = float((ref_logits - logits).abs().max().item())
        assert diff <= 1e-6, f"arm {arm}: dropout identity diff {diff}"


def _permute_atoms(batch, perm):
    n = int(perm.numel())
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    out = {k: v for k, v in batch.items()}
    out["atom_feat"] = batch["atom_feat"][perm]
    out["atom_scalar"] = batch["atom_scalar"][perm]
    out["atom_batch"] = batch["atom_batch"][perm]
    out["bond_index"] = inv[batch["bond_index"]]
    out["nbr_atom"] = inv[batch["nbr_atom"]]
    out["path_atom"] = inv[batch["path_atom"]]
    out["path_end"] = inv[batch["path_end"]]
    return out


def test_permutation_invariance(task, source, reference):
    _, _, base_init = reference
    batch = _batch(task, rows=(0,))
    gen = torch.Generator().manual_seed(4242)
    perm = torch.randperm(int(batch["atom_batch"].numel()), generator=gen)
    perm_batch = _permute_atoms(batch, perm)
    for arm in C.ARM_ORDER:
        if arm != "C":
            model, _ = _arm(task, base_init, source, arm)
            # randomise the residual so the new branch contributes
            with torch.no_grad():
                g = torch.Generator().manual_seed(11)
                for block in model.blocks:
                    block.fc2.weight.normal_(0.0, 0.05, generator=g)
                    block.fc2.bias.normal_(0.0, 0.05, generator=g)
        else:
            model, _ = _arm(task, base_init, source, arm)
        model.eval()
        with torch.no_grad():
            a = model(batch)
            b = model(perm_batch)
        diff = float((a - b).abs().max().item())
        assert diff <= 1e-4, f"arm {arm}: permutation diff {diff}"


def test_f_branch_is_adjacency_independent(task, source, reference):
    _, _, base_init = reference
    batch = _batch(task, rows=(0,))
    model, _ = _arm(task, base_init, source, "F")
    with torch.no_grad():
        gen = torch.Generator().manual_seed(3)
        h = torch.randn(int(batch["atom_batch"].numel()), C.BLOCK_WIDTH, generator=gen)
        base = model.propagate(h, batch)
        cut = dict(batch)
        cut["bond_index"] = batch["bond_index"][1:]
        cut["bond_feat"] = batch["bond_feat"][1:]
        other = model.propagate(h, cut)
    assert torch.equal(base, other)


def test_m_branch_responds_to_adjacency(task, source, reference):
    _, _, base_init = reference
    batch = _batch(task, rows=(0,))
    model, _ = _arm(task, base_init, source, "M")
    with torch.no_grad():
        gen = torch.Generator().manual_seed(3)
        for block in model.blocks:
            block.fc2.weight.normal_(0.0, 0.05, generator=gen)
            block.fc2.bias.normal_(0.0, 0.05, generator=gen)
        gen2 = torch.Generator().manual_seed(5)
        h = torch.randn(int(batch["atom_batch"].numel()), C.BLOCK_WIDTH, generator=gen2)
        base = model.propagate(h, batch)
        cut_batch = dict(batch)
        cut_batch["bond_index"] = batch["bond_index"][1:]
        cut_batch["bond_feat"] = batch["bond_feat"][1:]
        cut = model.propagate(h, cut_batch)
        # reverse-direction edges must duplicate the same messages exactly once
        doubled_batch = dict(batch)
        doubled_batch["bond_index"] = torch.cat([batch["bond_index"], batch["bond_index"].flip(1)], dim=0)
        doubled_batch["bond_feat"] = torch.cat([batch["bond_feat"], batch["bond_feat"]], dim=0)
        doubled = model.propagate(h, doubled_batch)
    assert float((base - cut).abs().max().item()) > 1e-7
    assert float((base - doubled).abs().max().item()) > 1e-7
    # one extra direction is not the same as the correct bidirectional mapping
    assert not torch.equal(base, doubled)


def test_v_context_is_graph_local(task, source, reference):
    _, _, base_init = reference
    batch = _batch(task, rows=(0, 1))
    model, _ = _arm(task, base_init, source, "V")
    model.eval()
    with torch.no_grad():
        a, b, scal = model._encode(batch)
        h = model.local(batch, a, b, scal)
        block = model.blocks[0]
        z = block.norm(h)
        ab = batch["atom_batch"]
        n_graphs = int(batch["n_graphs"])
        counts = z.new_zeros(n_graphs).index_add_(0, ab, torch.ones_like(ab, dtype=z.dtype))
        g = z.new_zeros((n_graphs, z.shape[1])).index_add_(0, ab, z) / counts.unsqueeze(1)
        manual = torch.stack([z[ab == i].mean(dim=0) for i in range(n_graphs)])
        assert float((g - manual).abs().max().item()) <= 1e-6
        z2 = z.clone()
        z2[ab == 1] = z2[ab == 1] + 100.0
        g2 = z2.new_zeros((n_graphs, z.shape[1])).index_add_(0, ab, z2) / counts.unsqueeze(1)
        assert torch.equal(g[0], g2[0])
        ctx_atoms = block.ctx_proj(g)[ab]
        for i in range(n_graphs):
            assert float((ctx_atoms[ab == i] - ctx_atoms[ab == i][0]).abs().max().item()) <= 1e-6


def _grad_norms(model):
    out = {}
    for name, p in model.named_parameters():
        if name.startswith(M.NEW_MODULE_PREFIXES) and p.grad is not None:
            out[name] = float(p.grad.norm().item())
    return out


def test_gradients_reach_new_modules_after_two_steps(task, source, reference):
    _, _, base_init = reference
    batch = _batch(task, rows=(0, 1, 2, 3, 4, 5, 6, 7))
    y = torch.as_tensor(task.labels[np.arange(8)], dtype=torch.float32)
    for arm in ("F", "M", "V"):
        model, _ = _arm(task, base_init, source, arm)
        optimizer, _ = M.make_optimizer_warm(model, source["optimizer_state_dict"])
        seen = {}
        for step in range(3):
            M.set_lrs(optimizer, phase=1, step=step)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = MET.masked_bce_loss(logits, y)
            loss.backward()
            seen[step] = _grad_norms(model)
            optimizer.step()
        late = {k: max(seen[1].get(k, 0.0), seen[2].get(k, 0.0)) for k in set(seen[1]) | set(seen[2])}
        assert all(seen[0].get(k, 0.0) == 0.0 for k in seen[0] if ".fc1." in k)
        assert all(v > 0 for k, v in late.items() if ".fc1." in k), arm
        assert all(v > 0 for k, v in late.items() if ".fc2." in k), arm
        if arm in ("M", "V"):
            assert late["bond_proj.weight"] > 0, arm
        if arm == "V":
            assert all(v > 0 for k, v in late.items() if ".ctx_proj." in k), arm


def test_scheduler_endpoints_and_monotonicity():
    audit = C.lr_audit()
    p = audit["phase12"]
    assert math.isclose(p["base_step0"], 5e-5, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["new_step0"], 3e-4, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["base_step40e"], 1e-5, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["new_step40e"], 3e-5, rel_tol=0, abs_tol=1e-15)
    base = [C.scheduled_lr(s, phase=1, kind="base") for s in range(0, C.TOTAL_STEPS_40, 997)]
    new = [C.scheduled_lr(s, phase=1, kind="new") for s in range(0, C.TOTAL_STEPS_40, 997)]
    assert all(a >= b for a, b in zip(base, base[1:]))
    assert all(a >= b for a, b in zip(new, new[1:]))
    assert all(b > a for a, b in zip(base, new))
    p3 = audit["phase3"]
    assert math.isclose(p3["step0"], 3e-5 + (3e-4 - 3e-5) / C.STEPS_PER_EPOCH, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p3["warmup_last"], 3e-4, rel_tol=0, abs_tol=1e-15)
    assert abs(p3["post_warmup_first"] - 3e-4) < 1e-6
    assert math.isclose(p3["step40e"], 1e-5, rel_tol=0, abs_tol=1e-15)
    warm = [C.scheduled_lr(s, phase=3, kind="all") for s in range(0, C.STEPS_PER_EPOCH)]
    assert all(a <= b for a, b in zip(warm, warm[1:]))
    post = [C.scheduled_lr(s, phase=3, kind="all") for s in range(C.STEPS_PER_EPOCH, C.TOTAL_STEPS_40, 1009)]
    assert all(a >= b for a, b in zip(post, post[1:]))
    assert warm[-1] == pytest.approx(3e-4, abs=1e-15)
    assert warm[0] > 3e-5 and warm[0] <= 3e-4


def test_masked_bce_is_the_historical_definition():
    torch.manual_seed(0)
    logits = torch.randn(17, 128)
    y = torch.randn(17, 128)
    y[torch.rand_like(y) < 0.4] = float("nan")
    got = MET.masked_bce_loss(logits, y)
    mask = torch.isfinite(y)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])
    assert torch.equal(got, ref)
    assert math.isfinite(float(got))


def test_test_isolation_and_order_plan(task):
    labels = np.load(C.LABELS_PATH, mmap_mode="r")
    assert labels.shape == (C.N_MOLECULES, C.N_TASKS)
    assert not np.isfinite(np.asarray(labels[task.test_rows])).any()
    tr = set(int(x) for x in task.train_rows)
    va = set(int(x) for x in task.valid_rows)
    te = set(int(x) for x in task.test_rows)
    assert not (tr & va) and not (tr & te) and not (va & te)
    plan1 = D.make_plan40(task)
    plan2 = D.make_plan40(task)
    assert len(plan1) == C.TOTAL_EPOCHS
    assert [p["row_order_sha256"] for p in plan1] == [p["row_order_sha256"] for p in plan2]
    summary = D.plan_summary(plan1, task)
    assert summary["isolation"]["passed"] is True
    assert plan1[0]["global_plan_sha256"] == plan2[0]["global_plan_sha256"]


def test_parameter_flags_and_shapes(task, source, reference):
    _, _, base_init = reference
    for arm in C.ARM_ORDER:
        model, _ = _arm(task, base_init, source, arm)
        assert all(p.requires_grad for p in model.parameters())
        assert next(model.parameters()).dtype == torch.float32
        assert float(model.drop.p) == 0.10
        batch = _batch(task, rows=(0,))
        logits = model(batch)
        assert logits.shape == (1, 128)
        assert torch.isfinite(logits).all()
