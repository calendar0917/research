"""Focused CPU tests for SDB-v0 (``sdb_v0`` / ``zinc_sdb_v0``).

Data-free: every test builds small toy topologies with
``tracks.ksvd.code.graph.from_edges``.  No ZINC data, no checkpoints, no
official test.  The ``check_*`` functions are also driven by the runner's
``sanity`` stage.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as runner


def _path_graph(n: int):
    return from_edges(n, [(index, index + 1) for index in range(n - 1)])


def _path_edges(n: int) -> dict[tuple[int, int], int]:
    return {(index, index + 1): 1 for index in range(n - 1)}


def _molecule(graph, types, edges, target=0.0) -> r2.MoleculeFeatures:
    phi = r2.build_phi(graph)
    marginal = r2.build_A(types, dict(edges), len(types))
    return r2.MoleculeFeatures(
        phi=phi.astype(np.float32),
        atom_idx=np.asarray(types, dtype=np.int64),
        A=marginal.astype(np.float32),
        n_nodes=int(len(types)),
        n_edges=int(len(edges)),
        y=float(target),
    )


def _relabel(graph, perm):
    n = len(perm)
    inverse = {int(old): int(new) for new, old in enumerate(perm)}
    edges = [(inverse[int(u)], inverse[int(v)]) for u, v in graph.edges()]
    return from_edges(n, edges)


def _module_source(module) -> str:
    return Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")


def _dict() -> np.ndarray:
    return sdb.random_normalized_dictionary(sdb.PHI_DIM, sdb.K_ATOMS, seed=123)


def _cd(molecule, D) -> np.ndarray:
    alpha = sdb.omp_codes(D, np.asarray(molecule.phi, dtype=np.float64), s=sdb.SPARSITY)
    q = r2.one_hot_q(molecule.atom_idx)
    return sdb.binding_tensor(alpha, q)


# ---------------------------------------------------------------------------
# checks (also run by the runner sanity stage)
# ---------------------------------------------------------------------------


def check_node_relabel_invariance() -> None:
    D = _dict()
    graph = _path_graph(5)
    types = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
    base = _molecule(graph, types, _path_edges(5))
    perm = [3, 0, 4, 1, 2]
    relabelled = _molecule(
        _relabel(graph, perm),
        np.asarray([types[perm[i]] for i in range(5)], dtype=np.int64),
        _path_edges(5),
    )
    assert np.allclose(_cd(base, D), _cd(relabelled, D), atol=1e-9)


def check_dictionary_input_chemistry_purity() -> None:
    D = _dict()
    graph = _path_graph(5)
    mol_a = _molecule(graph, np.asarray([0, 1, 2, 1, 0]), _path_edges(5))
    mol_b = _molecule(graph, np.asarray([3, 0, 4, 2, 1]), _path_edges(5))  # different chemistry
    alpha_a = sdb.omp_codes(D, np.asarray(mol_a.phi, dtype=np.float64), s=sdb.SPARSITY)
    alpha_b = sdb.omp_codes(D, np.asarray(mol_b.phi, dtype=np.float64), s=sdb.SPARSITY)
    assert np.array_equal(alpha_a, alpha_b)
    assert np.array_equal(np.asarray(mol_a.phi), np.asarray(mol_b.phi))
    body = _module_source(sdb)
    start, end = body.index("def codes_for_molecules"), body.index("def binding_matrices_for_molecules")
    segment = body[start:end]
    assert "atom_idx" not in segment and ".A" not in segment and "node_types" not in segment


def check_exact_sparsity() -> None:
    D = _dict()
    mol = _molecule(_path_graph(6), np.asarray([0, 1, 0, 2, 0, 1]), _path_edges(6))
    alpha = sdb.omp_codes(D, np.asarray(mol.phi, dtype=np.float64), s=sdb.SPARSITY)
    max_l0, within = sdb.exact_sparsity(alpha)
    assert within and max_l0 <= sdb.SPARSITY


def check_batching_invariance() -> None:
    D = _dict()
    mols = [
        _molecule(_path_graph(4), np.asarray([0, 1, 0, 0]), _path_edges(4)),
        _molecule(_path_graph(6), np.asarray([0, 2, 1, 0, 1, 0]), _path_edges(6)),
    ]
    codes = [sdb.omp_codes(D, np.asarray(m.phi, dtype=np.float64), s=sdb.SPARSITY) for m in mols]
    reference = [sdb.binding_tensor(a, r2.one_hot_q(m.atom_idx)) for a, m in zip(codes, mols)]
    alpha = torch.as_tensor(np.concatenate(codes, axis=0), dtype=torch.float64)
    q = torch.as_tensor(
        np.concatenate([r2.one_hot_q(m.atom_idx) for m in mols], axis=0), dtype=torch.float64
    )
    node_graph = torch.as_tensor(
        [i for i, m in enumerate(mols) for _ in range(m.n_nodes)], dtype=torch.long
    )
    c_matrix, _p = r2.segment_center_stats(alpha, q, node_graph, len(mols))
    batched = c_matrix.numpy()
    for i, ref in enumerate(reference):
        assert np.allclose(batched[i], ref, atol=1e-9)


def check_cd_assignment_sensitivity() -> None:
    D = _dict()
    edges = _path_edges(4)
    endpoint = _molecule(_path_graph(4), np.asarray([1, 0, 0, 0]), edges)
    internal = _molecule(_path_graph(4), np.asarray([0, 1, 0, 0]), edges)
    # same topology, same chemistry multiset (1 hetero + 3 carbon); only placement differs.
    assert np.array_equal(np.sort(endpoint.atom_idx), np.sort(internal.atom_idx))
    assert not np.allclose(_cd(endpoint, D), _cd(internal, D), atol=1e-9)


def check_chemistry_permutation() -> None:
    D = _dict()
    graph = _path_graph(5)
    mol = _molecule(graph, np.asarray([0, 1, 0, 2, 0]), _path_edges(5))
    alpha = sdb.omp_codes(D, np.asarray(mol.phi, dtype=np.float64), s=sdb.SPARSITY)
    q = r2.one_hot_q(mol.atom_idx)
    perm = np.asarray([2, 0, 4, 1, 3])
    q_perm = q[perm]
    alpha_perm = sdb.omp_codes(D, np.asarray(mol.phi, dtype=np.float64)[perm], s=sdb.SPARSITY)
    # dictionary codes follow the node permutation but chemistry marginal is unchanged
    assert np.array_equal(np.sort(mol.atom_idx), np.sort(mol.atom_idx[perm]))
    # structural marginal (column mean of phi) unchanged, chemistry marginal (q mean) unchanged
    assert np.allclose(np.asarray(mol.phi).mean(axis=0), np.asarray(mol.phi)[perm].mean(axis=0), atol=1e-6)
    assert np.allclose(q.mean(axis=0), q_perm.mean(axis=0), atol=1e-9)
    # C_D changes under a non-trivial chemistry permutation that breaks the alignment
    c_base = sdb.binding_tensor(alpha, q)
    c_perm = sdb.binding_tensor(alpha, q_perm)
    assert not np.allclose(c_base, c_perm, atol=1e-9)
    assert alpha_perm.shape == alpha.shape


def check_joint_permutation_invariance() -> None:
    D = _dict()
    graph = _path_graph(5)
    types = np.asarray([0, 1, 0, 2, 0], dtype=np.int64)
    base = _molecule(graph, types, _path_edges(5))
    perm = [3, 0, 4, 1, 2]
    relabelled = _molecule(
        _relabel(graph, perm),
        np.asarray([types[perm[i]] for i in range(5)], dtype=np.int64),
        _path_edges(5),
    )
    assert np.allclose(_cd(base, D), _cd(relabelled, D), atol=1e-9)


def check_no_raw_or_mixed_bypass() -> None:
    D = _dict()
    mol = _molecule(_path_graph(4), np.asarray([0, 1, 0, 0]), _path_edges(4))
    alpha = sdb.omp_codes(D, np.asarray(mol.phi, dtype=np.float64), s=sdb.SPARSITY)
    zero = np.zeros_like(alpha)
    q = r2.one_hot_q(mol.atom_idx)
    assert np.allclose(sdb.binding_tensor_from_codes(zero, q), 0.0, atol=1e-12)
    # 'binding_tensor' is defined purely as the centred product of its arguments.
    src = inspect.getsource(sdb.binding_tensor)
    assert "center_stats" in src


def check_no_official_test_access() -> None:
    for module in (sdb, runner):
        source = _module_source(module)
        assert "'test'" not in source and '"test"' not in source
        assert "_load_zinc" not in source
    tree = ast.parse(_module_source(runner))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value.strip().lower() != "test"


def check_binding_identity_exact() -> None:
    """If the dictionary spans phi exactly then C_phi == D C_D exactly."""
    graph = _path_graph(5)
    mol = _molecule(graph, np.asarray([0, 1, 0, 2, 0]), _path_edges(5))
    phi = np.asarray(mol.phi, dtype=np.float64)
    unique = np.unique(phi, axis=0)
    n = unique.shape[0]
    D = np.zeros((sdb.PHI_DIM, sdb.K_ATOMS), dtype=np.float64)
    norms = np.linalg.norm(unique, axis=1)
    norms[norms < 1e-12] = 1.0
    D[:, :n] = unique.T / norms[None, :]
    alpha = sdb.omp_codes(D, phi, s=min(sdb.SPARSITY, n))
    q = r2.one_hot_q(mol.atom_idx)
    c_phi = sdb.binding_tensor(phi, q)
    c_d = sdb.binding_tensor(alpha, q)
    residual = c_phi - D @ c_d
    assert np.linalg.norm(residual) / (np.linalg.norm(c_phi) + 1e-12) < 1e-5  # float32-stored phi


def check_iht_exact_sparsity() -> None:
    from tracks.ksvd.code import tccd_v0 as T

    D = torch.as_tensor(_dict(), dtype=torch.float32)
    X = torch.as_tensor(np.random.default_rng(0).standard_normal((24, sdb.PHI_DIM)), dtype=torch.float32)
    alpha = T.iht_codes(D, X, s=sdb.SPARSITY, steps=10).numpy()
    max_l0, within = sdb.exact_sparsity(alpha)
    assert within and max_l0 <= sdb.SPARSITY
    assert alpha.shape == (24, sdb.K_ATOMS)


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check",
    [
        check_node_relabel_invariance,
        check_dictionary_input_chemistry_purity,
        check_exact_sparsity,
        check_batching_invariance,
        check_cd_assignment_sensitivity,
        check_chemistry_permutation,
        check_joint_permutation_invariance,
        check_no_raw_or_mixed_bypass,
        check_no_official_test_access,
        check_binding_identity_exact,
        check_iht_exact_sparsity,
    ],
)
def test_sdb_v0_checks(check) -> None:
    check()


__all__ = [name for name in dir() if name.startswith("check_")]
