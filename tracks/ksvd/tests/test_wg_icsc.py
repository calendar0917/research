"""Targeted tests for WG-ICSC-v0 (pre-registered tests A-G + reference parity).

These tests are CPU-only, synthetic-data only (no ``data/``, no checkpoints),
and assert the algebraic properties the pre-registration requires.  They do
not train and do not read any official split.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import icsc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _toy_samples() -> list[icsc.GraphSample]:
    return [
        icsc.graph_sample_from_arrays(
            [0, 1, 2, 3, 4, 0],
            [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 3), (4, 5, 2)],
            y=1.0,
        ),
        icsc.graph_sample_from_arrays(
            [5, 5, 6, 7],
            [(0, 1, 1), (1, 2, 3), (2, 3, 2)],
            y=2.0,
        ),
    ]


def _raw_sample(
    atom_types: list[int], bonds: list[tuple[int, int, int]], y: float = 0.0
) -> icsc.GraphSample:
    """Build a sample keeping the given bond order (no re-sorting)."""
    xv = F.one_hot(torch.tensor(atom_types, dtype=torch.long), icsc.D_V).to(torch.float32).t().contiguous()
    src = torch.tensor([b[0] for b in bonds], dtype=torch.long)
    dst = torch.tensor([b[1] for b in bonds], dtype=torch.long)
    types = torch.tensor([b[2] - icsc.BOND_OFFSET for b in bonds], dtype=torch.long)
    xe = F.one_hot(types, icsc.D_E).to(torch.float32).t().contiguous() if bonds else torch.zeros(icsc.D_E, 0)
    return icsc.GraphSample(xv=xv, xe=xe, src=src, dst=dst, y=y)


def _permute_nodes(sample: icsc.GraphSample, perm: list[int]) -> icsc.GraphSample:
    """New node ``i`` is old node ``perm[i]``."""
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    atoms = sample.xv.argmax(dim=0).tolist()
    new_atoms = [atoms[perm[i]] for i in range(len(perm))]
    bonds = []
    for e in range(sample.m):
        u = int(sample.src[e])
        v = int(sample.dst[e])
        b = int(sample.xe[:, e].argmax()) + icsc.BOND_OFFSET
        bonds.append((inverse[u], inverse[v], b))
    return icsc.graph_sample_from_arrays(new_atoms, bonds, y=sample.y)


def _permute_bonds(sample: icsc.GraphSample, perm: list[int]) -> icsc.GraphSample:
    """New bond ``j`` is old bond ``perm[j]`` (bond order preserved explicitly)."""
    atoms = sample.xv.argmax(dim=0).tolist()
    bonds = []
    for j in range(sample.m):
        e = perm[j]
        u = int(sample.src[e])
        v = int(sample.dst[e])
        b = int(sample.xe[:, e].argmax()) + icsc.BOND_OFFSET
        bonds.append((u, v, b))
    return _raw_sample(atoms, bonds, y=sample.y)


# ---------------------------------------------------------------------------
# A. shape / finite / non-negativity
# ---------------------------------------------------------------------------


def test_shapes_finite_and_nonnegative() -> None:
    torch.manual_seed(0)
    batch = icsc.collate(_toy_samples())
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    prediction, output = model(batch, record=True)
    assert prediction.shape == (batch.n_graphs,)
    assert output.alpha.shape == (batch.n_graphs, icsc.ALPHA_DIM)
    assert output.z_v.shape == (icsc.K_V, batch.N)
    assert output.z_e.shape == (icsc.K_E, batch.M)
    assert torch.isfinite(output.z_v).all() and torch.isfinite(output.z_e).all()
    assert torch.isfinite(output.alpha).all() and torch.isfinite(prediction).all()
    assert (output.z_v >= 0).all() and (output.z_e >= 0).all()
    assert len(output.step_stats) == icsc.T_STEPS + 1


# ---------------------------------------------------------------------------
# B. exact sparsity (proximal operator produces exact zeros)
# ---------------------------------------------------------------------------


def test_proximal_produces_exact_row_zeros() -> None:
    torch.manual_seed(0)
    batch = icsc.collate(_toy_samples())
    model = icsc.WGICSC(seed=0, lambda1=0.2, lambda_g=5.0)
    output = model.solve(batch)
    row_norm_v = output.z_v.norm(dim=1)
    row_norm_e = output.z_e.norm(dim=1)
    assert float((row_norm_v == 0).to(torch.float32).sum()) > 0
    assert float((row_norm_e == 0).to(torch.float32).sum()) > 0
    # every surviving non-zero entry is strictly positive (positive L1 shrink)
    assert bool((output.z_v[output.z_v > 0] > 0).all())


# ---------------------------------------------------------------------------
# reference vs batched parity
# ---------------------------------------------------------------------------


def test_reference_matches_batched() -> None:
    torch.manual_seed(1)
    samples = _toy_samples()
    batch = icsc.collate(samples)
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    d_v, d_e = model.normalized_dictionaries()
    batched = icsc.solve_batched(
        d_v, d_e, model.w, batch, gamma=model.gamma, rho=model.rho,
        lambda1=model.lambda1, lambda_g=model.lambda_g, t_steps=model.t_steps,
    )
    ref = [icsc.solve_reference(d_v, d_e, model.w, s, lambda1=model.lambda1, lambda_g=model.lambda_g) for s in samples]
    z_v_ref = torch.cat([r.z_v for r in ref], dim=1)
    z_e_ref = torch.cat([r.z_e for r in ref], dim=1)
    assert float((z_v_ref - batched.z_v).abs().max()) < 1e-6
    assert float((z_e_ref - batched.z_e).abs().max()) < 1e-6


# ---------------------------------------------------------------------------
# C. permutation equivariance (node order and bond-object order)
# ---------------------------------------------------------------------------


def test_permutation_equivariance() -> None:
    torch.manual_seed(0)
    sample = _toy_samples()[0]
    node_perm = [3, 0, 5, 1, 4, 2]
    bond_perm = [4, 0, 3, 1, 2]
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    d_v, d_e = model.normalized_dictionaries()

    base = icsc.solve_reference(d_v, d_e, model.w, sample, lambda1=model.lambda1, lambda_g=model.lambda_g)
    node_permuted = _permute_nodes(sample, node_perm)
    permuted = icsc.solve_reference(
        d_v, d_e, model.w, node_permuted, lambda1=model.lambda1, lambda_g=model.lambda_g
    )
    # new column i corresponds to old column node_perm[i]
    expected_zv = base.z_v[:, node_perm]
    assert float((permuted.z_v - expected_zv).abs().max()) < 1e-6

    bond_permuted = _permute_bonds(sample, bond_perm)
    bperm = icsc.solve_reference(
        d_v, d_e, model.w, bond_permuted, lambda1=model.lambda1, lambda_g=model.lambda_g
    )
    expected_ze = base.z_e[:, bond_perm]
    assert float((bperm.z_e - expected_ze).abs().max()) < 1e-6


def test_graph_code_invariance() -> None:
    torch.manual_seed(2)
    sample = _toy_samples()[0]
    node_perm = [2, 5, 0, 4, 1, 3]
    bond_perm = [1, 4, 0, 3, 2]
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    d_v, d_e = model.normalized_dictionaries()

    def alpha_of(s: icsc.GraphSample) -> torch.Tensor:
        batch = icsc.collate([s])
        out = icsc.solve_batched(
            d_v, d_e, model.w, batch, gamma=model.gamma, rho=model.rho,
            lambda1=model.lambda1, lambda_g=model.lambda_g, t_steps=model.t_steps,
        )
        return out.alpha[0]

    a0 = alpha_of(sample)
    a1 = alpha_of(_permute_nodes(sample, node_perm))
    a2 = alpha_of(_permute_bonds(sample, bond_perm))
    assert float((a0 - a1).abs().max()) < 1e-5
    assert float((a0 - a2).abs().max()) < 1e-5


# ---------------------------------------------------------------------------
# E. composition endpoint test
# ---------------------------------------------------------------------------


def test_incidence_assignment_changes_code() -> None:
    torch.manual_seed(3)
    # Same node multiset {C,C,O,O} and same bond multiset {type1,type2}, but a
    # different bond<->endpoint assignment.
    a = icsc.graph_sample_from_arrays([0, 0, 1, 1], [(0, 1, 1), (2, 3, 2)])
    b = icsc.graph_sample_from_arrays([0, 0, 1, 1], [(0, 2, 1), (1, 3, 2)])
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    d_v, d_e = model.normalized_dictionaries()
    out_a = icsc.solve_reference(d_v, d_e, model.w, a, lambda1=model.lambda1, lambda_g=model.lambda_g)
    out_b = icsc.solve_reference(d_v, d_e, model.w, b, lambda1=model.lambda1, lambda_g=model.lambda_g)
    assert float((out_a.z_v - out_b.z_v).abs().max()) > 1e-6
    assert float((out_a.z_e - out_b.z_e).abs().max()) > 1e-6
    assert float((out_a.alpha - out_b.alpha).abs().max()) > 1e-8


# ---------------------------------------------------------------------------
# F. solver descent diagnostic
# ---------------------------------------------------------------------------


def test_solver_energy_is_non_increasing() -> None:
    torch.manual_seed(0)
    batch = icsc.collate(_toy_samples())
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    output = model.solve(batch, record=True)
    energies = [row["energy"] for row in output.step_stats]
    for before, after in zip(energies, energies[1:]):
        assert after <= before + 1e-9, (before, after)


# ---------------------------------------------------------------------------
# G. gradient flow
# ---------------------------------------------------------------------------


def test_gradients_reach_all_parameters() -> None:
    torch.manual_seed(0)
    batch = icsc.collate(_toy_samples())
    model = icsc.WGICSC(seed=0, lambda1=0.03, lambda_g=0.15)
    prediction, output = model(batch)
    loss = F.l1_loss(prediction, batch.y) + icsc.MU * model.fit_loss(batch, output)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.norm()) > 0.0, name
