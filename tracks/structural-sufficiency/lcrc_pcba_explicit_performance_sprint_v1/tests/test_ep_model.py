"""Model-definition tests: A0 parity, zero residuals, A1 geometry, gradient flow."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ep_common as C
import ep_data as D
import ep_model as M
import nmp_model as NM
import pc_metrics as MET

_TASK = None
_BASE = None


def _task():
    global _TASK, _BASE
    if _TASK is None:
        _TASK = D.load_task()
        _BASE = NM.build_reference(_TASK.spec, _TASK.norm, C.N_TASKS, 0)
    return _TASK, _BASE


def _synthetic_features(task, modules, rows: np.ndarray, seed: int = 7) -> D.Features:
    """Deterministic, correctly aligned fake features for module-level tests."""
    mods = tuple(modules)
    rng = np.random.default_rng(seed)
    feat = D.Features(modules=mods)
    rows = np.asarray(rows, dtype=np.int64)
    if "A2" in mods:
        matrix = np.full((C.N_MOLECULES, C.PHYSCHEM_DIM), np.nan, dtype=np.float32)
        matrix[rows] = rng.normal(size=(len(rows), C.PHYSCHEM_DIM)).astype(np.float32)
        matrix[rows, -1] = 0.0
        feat.physchem = {"matrix": matrix, "feature_hash": "synthetic", "norm_fingerprint": "synthetic",
                         "meta": {"files": {"cache_sha256": "synthetic"}, "algorithm": C.PHYSCHEM_ALGO,
                                  "rdkit_version": "synthetic", "built_rows": len(rows), "failed_rows": 0,
                                  "coverage_train": 1.0, "coverage_valid": 1.0}}
    if "A3" in mods:
        atom_ptr = np.asarray(task.cache["atom_ptr"], dtype=np.int64)
        rows_sorted = np.sort(rows)
        counts = np.zeros(int(atom_ptr[-1]), dtype=np.int32)
        row_counts = np.zeros(C.N_MOLECULES, dtype=np.int64)
        for r in rows_sorted:
            a0, a1 = int(atom_ptr[r]), int(atom_ptr[r + 1])
            c = rng.integers(1, 5, size=a1 - a0).astype(np.int32)
            counts[a0:a1] = c
            row_counts[r] = int(c.sum())
        ptr = np.zeros(C.N_MOLECULES + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(row_counts)
        tokens = rng.integers(0, C.N_BRANCH_BUCKETS, size=int(ptr[-1])).astype(np.int16)
        feat.branch = {"row_token_ptr": ptr, "token_bucket": tokens, "atom_token_count": counts,
                       "feature_hash": "synthetic", "meta": {"files": {"cache_sha256": "synthetic"},
                       "algorithm": C.BRANCH_ALGO, "n_buckets": C.N_BRANCH_BUCKETS,
                       "n_tokens": int(tokens.size), "collision_stats": {}}}
    return feat


def _build(arm: str, device: str = "cpu", seed: int = 0):
    task, (_, _, base_init) = _task()
    mods = C.arm_modules(arm)
    model, audit = M.build_model(
        arm, task.spec, task.norm, C.N_TASKS, seed=seed, device=device,
        base_init_state=base_init, combine=",".join(mods) if arm == "COMBO" else "",
    )
    return model, audit


def test_a0_state_is_exactly_the_historical_a_model():
    task, (ref, _, base_init) = _task()
    hist, _ = NM.build_arm("A", task.spec, task.norm, C.N_TASKS, seed=0, device="cpu",
                           base_init_state=base_init, zero_residual=True)
    model, audit = _build("A0")
    hist_state = hist.state_dict()
    got = model.state_dict()
    assert set(got) == set(hist_state)
    for key, value in hist_state.items():
        assert torch.equal(value, got[key]), key
    assert audit["total_parameters"] == C.HISTORICAL_A0["parameters"]
    # At initialisation A0 reproduces C1 exactly (its residual head is zero).
    rows = task.train_rows[:8]
    batch = D.augmented_collate(task, rows, D.load_features(task, ()))
    model.eval()
    ref.eval()
    with torch.no_grad():
        assert float((model(batch) - ref(batch)).abs().max()) == 0.0


def test_shared_c1_and_original_readout_identical_across_arms():
    task, _ = _task()
    models = {}
    for arm in ("A0", "A1", "A2", "A3"):
        models[arm], _ = _build(arm)
    ref_state = models["A0"].state_dict()
    base_keys = [k for k in ref_state if not k.startswith(M.EXTRA_PREFIXES)]
    assert base_keys, "base keys missing"
    for arm, model in models.items():
        state = model.state_dict()
        for k in base_keys:
            assert torch.equal(state[k], ref_state[k]), f"{arm} {k}"
        extra = [k for k in state if k.startswith(M.EXTRA_PREFIXES)]
        assert (len(extra) > 0) == (arm != "A0")
    # C1 base fingerprint (readout excluded) equals the frozen historical value.
    for arm, model in models.items():
        fp = C.state_fingerprint(
            {k: v for k, v in model.state_dict().items()
             if not k.startswith(M.EXTRA_PREFIXES) and not k.startswith(("blocks.", "readout."))}
        )
        assert fp == C.HISTORICAL_A0["base_init_fingerprint"]


def test_zero_residual_identity_all_arms():
    task, (ref, _, _) = _task()
    a0, _ = _build("A0")
    rows = task.train_rows[:8]
    feats0 = D.load_features(task, ())
    batch0 = D.augmented_collate(task, rows, feats0)
    a0.eval()
    ref.eval()
    with torch.no_grad():
        base_logits = a0(batch0)
    for arm in ("A1", "A2", "A3"):
        mods = C.arm_modules(arm)
        model, _ = _build(arm)
        model.eval()
        feats = _synthetic_features(task, mods, rows)
        batch = D.augmented_collate(task, rows, feats)
        with torch.no_grad():
            logits = model(batch)
        assert tuple(logits.shape) == (len(rows), C.N_TASKS)
        assert float((logits - base_logits).abs().max()) == 0.0, arm


def test_extra_initialisation_does_not_touch_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    for arm in ("A0", "A1", "A2", "A3"):
        _build(arm)
    assert torch.equal(before, torch.get_rng_state())


def test_a1_attention_normalises_per_molecule_task_and_head():
    task, _ = _task()
    rows = task.train_rows[:6]
    model, _ = _build("A1")
    model.eval()
    batch = D.augmented_collate(task, rows, _synthetic_features(task, ("A1",), rows))
    a, b, scal = model._encode(batch)
    h = model.drop(model.local(batch, a, b, scal))
    with torch.no_grad():
        z = model.tc.norm(h)
        key = model.tc.key_proj(z)
        scores = torch.einsum("nd,tkd->ntk", key, model.tc.query) / (64 ** 0.5)
        alpha = M.segment_softmax(scores.reshape(-1, C.N_TASKS * M.N_QUERIES), batch["atom_batch"], len(rows))
        sums = []
        for g in range(len(rows)):
            sums.append(alpha[batch["atom_batch"] == g].sum(0).numpy())
    dev = float(np.abs(np.stack(sums) - 1.0).max())
    assert dev <= 1e-5, dev


def test_a1_is_permutation_invariant_and_per_molecule():
    task, _ = _task()
    rows = task.train_rows[:4]
    model, _ = _build("A1")
    # At initialisation w=0 makes the residual identically zero, so give the
    # readout a non-trivial output map before testing its geometry.
    with torch.no_grad():
        model.tc.w.normal_(0.0, 0.1)
        model.tc.b.normal_(0.0, 0.1)
    model.eval()
    batch = D.augmented_collate(task, rows, _synthetic_features(task, ("A1",), rows))
    a, b, scal = model._encode(batch)
    h = model.drop(model.local(batch, a, b, scal))
    n_graphs = len(rows)
    with torch.no_grad():
        base = model.tc(h, batch["atom_batch"], n_graphs)
        perm = torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(3))
        permuted = model.tc(h[perm], batch["atom_batch"][perm], n_graphs)
        assert float((base - permuted).abs().max()) <= 1e-6
        # Perturb only graph 0: other graphs' task-conditioned residuals are unchanged.
        h2 = h.clone()
        h2[batch["atom_batch"] == 0] += 5.0
        other = model.tc(h2, batch["atom_batch"], n_graphs)
        assert float((base[1:] - other[1:]).abs().max()) <= 1e-6
        assert float((base[0] - other[0]).abs().max()) > 0.0


def test_new_modules_do_not_need_graph_tensors():
    task, _ = _task()
    rows = task.train_rows[:4]
    for arm, mods in (("A1", ("A1",)), ("A2", ("A2",)), ("A3", ("A3",))):
        model, _ = _build(arm)
        model.eval()
        feats = _synthetic_features(task, mods, rows)
        batch = D.augmented_collate(task, rows, feats)
        a, b, scal = model._encode(batch)
        h = model.drop(model.local(batch, a, b, scal))
        allowed = {"atom_batch": batch["atom_batch"], "n_graphs": batch["n_graphs"]}
        with torch.no_grad():
            model.propagate(h, allowed)  # original A readout: h/atom_batch only
            if "A1" in mods:
                model.tc(h, batch["atom_batch"], len(rows))
            if "A2" in mods:
                model.pc(batch["physchem"])
            if "A3" in mods:
                model.bt(h, batch["bt_ptr"], batch["bt_bucket"])
        state = model.state_dict()
        assert not any(k.startswith(("blocks.",)) for k in state)


@pytest.mark.parametrize("arm", ["A1", "A2", "A3"])
def test_extra_modules_receive_gradients_after_three_steps(arm):
    task, _ = _task()
    rows = task.train_rows[:16]
    model, _ = _build(arm)
    feats = _synthetic_features(task, C.arm_modules(arm), rows, seed=11)
    opt, _ = NM.make_optimizer_scratch(model)
    model.train()
    for step in range(3):
        batch = D.augmented_collate(task, rows, feats)
        y = torch.from_numpy(np.asarray(task.labels[rows], dtype=np.float32))
        NM.set_lrs(opt, step=step)
        opt.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch), y)
        loss.backward()
        opt.step()
    final = {n: (0.0 if p.grad is None else float(p.grad.norm())) for n, p in model.named_parameters()}
    extra = {n: v for n, v in final.items() if n.startswith(M.EXTRA_PREFIXES)}
    assert extra
    assert all(np.isfinite(v) for v in extra.values())
    zero = [n for n, v in extra.items() if v == 0.0]
    assert not zero, f"extra parameters without gradient after 3 steps: {zero}"


def test_combo_module_composition_matches_single_model_definitions():
    task, _ = _task()
    a1, _ = _build("A1")
    a3, _ = _build("A3")
    taskm, (_, _, base_init) = _task()
    combo, audit = M.build_model(
        "COMBO", taskm.spec, taskm.norm, C.N_TASKS, seed=0, device="cpu",
        base_init_state=base_init, combine="A1,A3",
    )
    assert audit["modules"] == ["A1", "A3"]
    assert audit["total_parameters"] == C.expected_parameters(("A1", "A3"))
    st1, st3, stc = a1.state_dict(), a3.state_dict(), combo.state_dict()
    for key in st1:
        if key.startswith("tc.") or (not key.startswith(M.EXTRA_PREFIXES)):
            assert torch.equal(st1[key], stc[key]), key
    for key in st3:
        if key.startswith("bt."):
            assert torch.equal(st3[key], stc[key]), key
