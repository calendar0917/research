"""CPU correctness tests for the Cross-atom Predictive Evidence arms.

The load-bearing claims are: exactly 1,027,732 parameters in both arms, a
bitwise-identical original W2 parameter subset and initial output, identical
new tensors in P and S, a genuine *different-atom* second-order statistic that
equals the explicit ``i != j`` double loop, exact zero for ``n == 1``, no
cross-molecule interaction, one pass of every original module, one dropout call,
and no iterative propagation / attention / extra width anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import cpe_common as C
import cpe_model as M
import bs_model as BSM

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# reference implementations (explicit loops, deliberately naive)
# ---------------------------------------------------------------------------


def reference_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    out = np.zeros(a.shape[1], dtype=np.float64)
    if n < 2:
        return out
    for i in range(n):
        for j in range(n):
            if i != j:
                out += a[i].astype(np.float64) * b[j].astype(np.float64)
    return out / float(n * (n - 1))


def reference_self(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    if n == 0:
        return np.zeros(a.shape[1], dtype=np.float64)
    return (a.astype(np.float64) * b.astype(np.float64)).sum(axis=0) / float(n)


def stats_of(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    at = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    bt = torch.as_tensor(np.asarray(b), dtype=torch.float32)
    st = M.pair_statistics(at, bt, torch.zeros(at.shape[0], dtype=torch.long), 1)
    return {
        "cross": st["pair_cross"][0].numpy().astype(np.float64),
        "self": st["pair_self"][0].numpy().astype(np.float64),
        "mean_A": st["mean_A"][0].numpy().astype(np.float64),
        "mean_B": st["mean_B"][0].numpy().astype(np.float64),
    }


# ---------------------------------------------------------------------------
# parameters / init
# ---------------------------------------------------------------------------


def test_parameter_counts_are_exactly_1027732(models):
    for key in C.MODEL_ORDER:
        model, audit = models[key]
        assert int(sum(p.numel() for p in model.parameters())) == 1_027_732
        assert audit["new_parameters"] == 33_408
        assert audit["base_parameters"] == 994_324
        assert audit["total_parameters"] == 1_027_732
        evidence_params = {n: p for n, p in model.named_parameters() if n.startswith("evidence.")}
        assert sum(p.numel() for p in evidence_params.values()) == 33_408
        assert set(evidence_params) == {
            "evidence.linear_A.weight",
            "evidence.linear_A.bias",
            "evidence.linear_B.weight",
            "evidence.linear_B.bias",
            "evidence.alpha_A",
            "evidence.alpha_B",
            "evidence.alpha_pair",
        }
    assert C.NEW_PARAMETERS == 33_408
    assert C.EXPECTED_TOTALS == {"P": 1_027_732, "S": 1_027_732}
    assert C.NEW_BRANCH["linear_A"]["parameters"] == 16_512
    assert C.NEW_BRANCH["linear_B"]["parameters"] == 16_512


def test_original_w2_subset_bitwise_equal(models):
    base_state = models["W2"][0].state_dict()
    for key in C.MODEL_ORDER:
        state = models[key][0].state_dict()
        for name, value in base_state.items():
            assert name in state, (key, name)
            assert torch.equal(value.cpu(), state[name].cpu()), (key, name)
        assert models[key][1]["original_w2_subset"]["all_bitwise_equal"]


def test_new_tensors_identical_between_arms_and_zero_alphas(models):
    p_state = models["P"][0].state_dict()
    s_state = models["S"][0].state_dict()
    new_keys = models["P"][1]["new_state_keys"]
    assert new_keys == models["S"][1]["new_state_keys"]
    assert len(new_keys) == 7
    for name in new_keys:
        assert torch.equal(p_state[name].cpu(), s_state[name].cpu()), name
    for key in C.MODEL_ORDER:
        ev = models[key][0].evidence
        assert torch.count_nonzero(ev.alpha_A).item() == 0
        assert torch.count_nonzero(ev.alpha_B).item() == 0
        assert torch.count_nonzero(ev.alpha_pair).item() == 0
    # full state dicts are bitwise identical: the arms differ only in pair_mode
    for name in p_state:
        assert torch.equal(p_state[name].cpu(), s_state[name].cpu()), name
    assert models["P"][0].evidence.pair_mode == "cross"
    assert models["S"][0].evidence.pair_mode == "self"


def test_new_module_construction_does_not_advance_global_rng(spec, norm):
    """Only the *new* module is RNG-neutral; the historical W2 builder is not
    (its constructor consumes RNG before the frozen trunk state overwrites it,
    exactly as in the historical runs, and the trainer calls
    ``seed_everything`` afterwards)."""
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=0, device="cpu")
    torch.manual_seed(12345)
    before = torch.get_rng_state().clone()
    for key in C.MODEL_ORDER:
        M.build_model(key, spec, norm, C.N_TASKS, seed=0, device="cpu", base_model=base)
        assert torch.equal(before, torch.get_rng_state()), key


def test_initial_logits_bitwise_equal_to_w2(models, batch):
    with torch.no_grad():
        models["W2"][0].eval()
        ref = models["W2"][0](batch)
        for key in C.MODEL_ORDER:
            model = models[key][0]
            model.eval()
            got = model(batch)
            assert torch.equal(ref, got), key
            assert float((ref - got).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# the statistic itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 9])
def test_cross_equals_explicit_i_ne_j_double_loop(n):
    rng = np.random.default_rng(20261108 + n)
    a = rng.normal(size=(n, 7)).astype(np.float32)
    b = rng.normal(size=(n, 7)).astype(np.float32)
    got = stats_of(a, b)
    assert np.abs(got["cross"] - reference_cross(a, b)).max() <= 1e-6
    assert np.abs(got["self"] - reference_self(a, b)).max() <= 1e-6


def test_cross_is_not_the_same_statistic_as_self():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 5)).astype(np.float32)
    b = rng.normal(size=(4, 5)).astype(np.float32)
    got = stats_of(a, b)
    assert np.abs(got["cross"] - got["self"]).max() > 1e-3


def test_ab_swap_leaves_second_order_statistic_unchanged():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(6, 5)).astype(np.float32)
    b = rng.normal(size=(6, 5)).astype(np.float32)
    ab = stats_of(a, b)
    ba = stats_of(b, a)
    assert np.abs(ab["cross"] - ba["cross"]).max() == 0.0
    assert np.abs(ab["self"] - ba["self"]).max() == 0.0
    # the first-order terms are exchanged, not preserved
    assert np.abs(ab["mean_A"] - ba["mean_B"]).max() == 0.0
    assert np.abs(ab["mean_B"] - ba["mean_A"]).max() == 0.0


def test_single_atom_graph_has_exactly_zero_cross():
    a = np.array([[0.5, -0.25, 1.0]], dtype=np.float32)
    b = np.array([[0.25, 0.75, -1.0]], dtype=np.float32)
    at = torch.as_tensor(a)
    bt = torch.as_tensor(b)
    st = M.pair_statistics(at, bt, torch.zeros(1, dtype=torch.long), 1)
    assert float(st["pair_cross"].abs().max().item()) == 0.0
    assert float(st["pair_self"][0, 0].item()) == pytest.approx(0.5 * 0.25)
    assert st["n_single_atom_graphs"] == 1


def test_empty_graph_is_safe_and_reported():
    empty = torch.zeros((0, 4))
    st = M.pair_statistics(empty, empty, torch.zeros(0, dtype=torch.long), 3)
    assert st["n_empty_graphs"] == 3
    assert torch.count_nonzero(st["pair_cross"]).item() == 0
    assert torch.count_nonzero(st["pair_self"]).item() == 0
    assert torch.count_nonzero(st["mean_A"]).item() == 0
    assert torch.isfinite(st["pair_cross"]).all()
    assert torch.isfinite(st["mean_A"]).all()


def test_two_atom_cooccurrence_is_captured_by_cross_only():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    a[0, 0] = 1.0  # evidence A only on atom 0
    b[1, 0] = 1.0  # evidence B only on atom 1
    got = stats_of(a, b)
    assert got["cross"][0] == pytest.approx(0.5)  # mean over the 2 ordered pairs
    assert got["self"][0] == 0.0
    # different task: no co-occurrence at all in either statistic
    assert got["cross"][1] == 0.0 and got["self"][1] == 0.0


def test_no_cross_molecule_interaction_in_a_batch():
    rng = np.random.default_rng(3)
    sizes = [4, 2, 3, 1]
    a = rng.normal(size=(sum(sizes), 5)).astype(np.float32)
    b = rng.normal(size=(sum(sizes), 5)).astype(np.float32)
    groups = torch.as_tensor([g for g, n in enumerate(sizes) for _ in range(n)], dtype=torch.long)
    joint = M.pair_statistics(torch.as_tensor(a), torch.as_tensor(b), groups, len(sizes))
    off = 0
    for g, n in enumerate(sizes):
        single = M.pair_statistics(
            torch.as_tensor(a[off : off + n]),
            torch.as_tensor(b[off : off + n]),
            torch.zeros(n, dtype=torch.long),
            1,
        )
        for key in ("pair_cross", "pair_self", "mean_A", "mean_B"):
            assert torch.equal(joint[key][g], single[key][0]), (g, key)
        off += n


# ---------------------------------------------------------------------------
# model-level architecture
# ---------------------------------------------------------------------------


def test_evidence_module_is_only_linear_and_coefficients(models):
    types = [type(m).__name__ for name, m in models["P"][0].evidence.named_modules() if name]
    assert types == ["Linear", "Linear"]
    names = set(dict(models["P"][0].evidence.named_parameters()))
    assert names == {
        "linear_A.weight",
        "linear_A.bias",
        "linear_B.weight",
        "linear_B.bias",
        "alpha_A",
        "alpha_B",
        "alpha_pair",
    }
    assert tuple(models["P"][0].evidence.linear_A.weight.shape) == (128, 128)
    assert tuple(models["P"][0].evidence.linear_B.weight.shape) == (128, 128)
    for name in ("alpha_A", "alpha_B", "alpha_pair"):
        assert tuple(getattr(models["P"][0].evidence, name).shape) == (128,)


def test_single_pass_of_every_original_module(models, batch):
    for key in C.MODEL_ORDER:
        counts = M.count_module_calls(models[key][0], batch, training=False)
        assert counts["atom_encoder"] == 1, (key, counts)
        assert counts["bond_encoder"] == 1, (key, counts)
        assert counts["local"] == 1, (key, counts)
        assert counts["f"] == 1, (key, counts)
        assert counts["rho"] == 1, (key, counts)
        assert counts["readout"] == 1, (key, counts)
        assert counts["dropout"] == 1, (key, counts)
        assert counts["evidence"] == 1, (key, counts)
        assert counts["evidence_linear_A"] == 1 and counts["evidence_linear_B"] == 1


def test_post_local_path_reads_no_structure(models, batch):
    keys_to_drop = (
        "bond_index",
        "bond_feat",
        "bond_type",
        "bond_batch",
        "nbr_atom",
        "nbr_code",
        "path_atom",
        "path_cat",
        "path_end63",
        "path_end",
        "path_closure",
        "path_graph",
    )
    stripped = {k: v for k, v in batch.items() if k not in keys_to_drop}
    for key in C.MODEL_ORDER:
        model = models[key][0]
        model.eval()
        _a, _b, _scal, h = model.encode_h(batch)
        with torch.no_grad():
            full = model.post_local(h, batch) + model.evidence(h, batch["atom_batch"], int(batch["n_graphs"]))
            part = model.post_local(h, stripped) + model.evidence(
                h, stripped["atom_batch"], int(stripped["n_graphs"])
            )
        assert torch.equal(full, part), key


def test_graph_logits_permutation_invariant(models, batch):
    permuted, _p = BSM.permute_atoms(batch, seed=20261108)
    with torch.no_grad():
        for key in C.MODEL_ORDER:
            model = models[key][0]
            model.eval()
            a = model(batch)
            b = model(permuted)
            assert float((a - b).abs().max().item()) <= 2e-5, key


def test_evidence_outputs_and_residual_are_finite(models, batch):
    for key in C.MODEL_ORDER:
        model = models[key][0]
        model.eval()
        with torch.no_grad():
            logits, aux = model(batch, return_aux=True)
        ev = aux["evidence"]
        assert torch.isfinite(logits).all(), key
        assert torch.isfinite(ev["evidence_A"]).all()
        assert torch.isfinite(ev["pair_cross"]).all()
        assert torch.isfinite(ev["pair_self"]).all()
        assert ev["evidence_A"].abs().max().item() <= 1.0
        assert ev["evidence_B"].abs().max().item() <= 1.0
        # alphas are zero at init -> residual is exactly zero and logits == base
        assert float(ev["residual"].abs().max().item()) == 0.0
        assert torch.equal(logits, aux["base_logits"])


def test_evidence_diagnostics_shape_and_content(models, batch):
    for key in C.MODEL_ORDER:
        diag = models[key][0].evidence_diagnostics(batch)
        assert diag["pair_mode"] == C.PAIR_MODES[key]
        assert diag["finite"] is True
        assert diag["n_empty_graphs"] == 0
        assert diag["alpha"]["pair_l2"] == 0.0
        assert diag["magnitudes"]["pair_residual_abs_mean"] == 0.0
        assert diag["magnitudes"]["main_effect_residual_abs_mean"] == 0.0
        assert diag["statistics"]["pair_cross_abs_mean"] >= 0.0
        assert diag["statistics"]["pair_self_abs_mean"] >= 0.0
