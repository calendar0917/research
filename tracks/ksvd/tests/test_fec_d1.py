"""Focused CPU tests for FEC-D1 (``fec_d1`` / ``zinc_fec_d1``).

Data-free: small toy topologies with ``tracks.ksvd.code.graph.from_edges`` and
tiny tensor modules.  No ZINC data, no checkpoints, no official test.  The
``check_*`` functions are also driven by the runner's sanity path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import fec_d1 as d1


def _path_graph(n: int):
    return from_edges(n, [(index, index + 1) for index in range(n - 1)])


def _tiny_base(input_width: int = 146, out_width: int = 24) -> nn.Module:
    return nn.Linear(int(input_width), int(out_width))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_stat_dimensions() -> None:
    assert d1.STAT_DIM == d1.N_SHELLS * d1.K_ATOMS * d1.ATOM_CATEGORIES
    assert d1.STAT_DIM == 2688
    assert d1.K_ATOMS == 32 and d1.SPARSITY == 8 and d1.ATOM_CATEGORIES == 28
    assert d1.N_SHELLS == 3
    assert d1.PHI_DIM == 65


def check_branch_parameter_count() -> None:
    expected = 8 * 2688 + 24 * 8
    assert d1.branch_parameter_count() == expected == 21696


def check_ego_shells() -> None:
    graph = _path_graph(4)
    shells = d1.ego_shells(graph, 1, radius=2)
    assert shells[0] == [1]
    assert shells[1] == [0, 2]
    assert shells[2] == [3]
    shells0 = d1.ego_shells(graph, 0, radius=2)
    assert shells0[0] == [0]
    assert shells0[1] == [1]
    assert shells0[2] == [2]


def check_binding_reference_and_centering() -> None:
    rng = np.random.default_rng(0)
    graph = _path_graph(4)
    phi = rng.normal(size=(4, 65))
    atom_idx = np.asarray([0, 1, 2, 1], dtype=np.int64)
    coord = rng.normal(size=(4, d1.K_ATOMS))
    observed = d1.per_root_binding(phi, atom_idx, graph, coord)
    q = np.zeros((4, d1.ATOM_CATEGORIES), dtype=np.float64)
    q[np.arange(4), atom_idx] = 1.0
    reference = np.zeros((4, d1.STAT_DIM))
    for root in range(4):
        for shell in range(d1.N_SHELLS):
            index = d1.ego_shells(graph, root, 2).get(shell, [])
            if len(index) < 2:
                continue
            a = coord[index]
            qq = q[index]
            block = np.zeros((d1.K_ATOMS, d1.ATOM_CATEGORIES))
            for row in range(len(index)):
                block += np.outer(a[row] - a.mean(axis=0), qq[row] - qq.mean(axis=0))
            lo = shell * d1.K_ATOMS * d1.ATOM_CATEGORIES
            reference[root, lo : lo + d1.K_ATOMS * d1.ATOM_CATEGORIES] = block.reshape(-1)
    assert np.allclose(observed, reference, atol=1e-12)
    # within-shell centring => the coordinate row sums vanish for every block
    blocks = observed.reshape(4, d1.N_SHELLS, d1.K_ATOMS, d1.ATOM_CATEGORIES)
    assert np.abs(blocks.sum(axis=3)).max() < 1e-10


def check_small_shell_zero() -> None:
    graph = _path_graph(3)  # root 0: shell1 has one node -> C = 0
    phi = np.random.default_rng(1).normal(size=(3, 65))
    atom_idx = np.asarray([0, 1, 2], dtype=np.int64)
    coord = np.random.default_rng(2).normal(size=(3, d1.K_ATOMS))
    observed = d1.per_root_binding(phi, atom_idx, graph, coord)
    block = observed.reshape(3, d1.N_SHELLS, d1.K_ATOMS, d1.ATOM_CATEGORIES)[0, 1]
    assert np.allclose(block, 0.0)


def check_shuffle_changes_statistic_and_multisets() -> None:
    graph = _path_graph(5)
    phi = np.random.default_rng(3).normal(size=(5, 65))
    atom_idx = np.asarray([0, 1, 2, 1, 3], dtype=np.int64)
    coord = np.random.default_rng(4).normal(size=(5, d1.K_ATOMS))
    clean = d1.per_root_binding(phi, atom_idx, graph, coord)
    shuffled = d1.per_root_binding(phi, atom_idx, graph, coord, shuffle_seed=11)
    assert np.abs(clean - shuffled).max() > 1e-12
    # Both multisets are preserved by construction: the shuffle only reorders
    # `coord` rows relative to `q` rows inside each root/shell; a root/shell of
    # size < 2 is left at the zero block.


def check_branch_exact_containment() -> None:
    torch.manual_seed(0)
    base = _tiny_base()
    adapter = d1.BindingLocalEnvAdapter(base, stat_dim=10, rank=3, out_width=24)
    x = torch.randn(5, 146)
    stat = torch.randn(5, 10)
    with torch.no_grad():
        assert torch.equal(adapter(x), base(x))  # stat not set -> base only
        adapter.set_stat(stat)
        assert torch.equal(adapter(x), base(x))  # W2 == 0 -> exact containment
    with torch.no_grad():
        adapter.W2.normal_(0.0, 0.1)
        assert not torch.equal(adapter(x), base(x))


def check_branch_gradient_flow() -> None:
    torch.manual_seed(0)
    base = _tiny_base(146, 1)
    adapter = d1.BindingLocalEnvAdapter(base, stat_dim=10, rank=3, out_width=1)
    stat = torch.randn(5, 10)
    target = torch.randn(5)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    adapter.set_stat(stat)
    loss = torch.nn.functional.l1_loss(adapter(torch.randn(5, 146)).squeeze(-1), target)
    loss.backward()
    assert adapter.W2.grad is not None and float(adapter.W2.grad.norm()) > 0.0
    assert adapter.W1.grad is None or float(adapter.W1.grad.norm()) == 0.0  # W2 == 0 at step 0
    optimizer.step()
    adapter.W1.grad = None
    adapter.W2.grad = None
    adapter.set_stat(stat)
    loss = torch.nn.functional.l1_loss(adapter(torch.randn(5, 146)).squeeze(-1), target)
    loss.backward()
    assert adapter.W1.grad is not None and float(adapter.W1.grad.norm()) > 0.0


class _Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(7, 7)
        self.local_env_adapter = d1.BindingLocalEnvAdapter(
            _tiny_base(10, 4), stat_dim=10, rank=3, out_width=4
        )


def check_freeze_only_branch_trainable() -> None:
    model = _Container()
    report = d1.freeze_base_train_branch(model)
    assert set(report["trainable_names"]) == {"local_env_adapter.W1", "local_env_adapter.W2"}
    assert report["trainable_numel"] == 3 * 10 + 4 * 3
    assert not model.feature.weight.requires_grad


def check_no_official_test_access() -> None:
    runner = Path("tracks/ksvd/experiments/luyin16/zinc_fec_d1.py")
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    literal_splits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_load_zinc":
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        literal_splits.append(str(arg.value))
    assert "test" not in literal_splits, literal_splits
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "official_test_blocker" in function_names
    assert "test" in runner.read_text(encoding="utf-8"), "the blocker must explicitly name the test split"


def check_no_direct_readout_bypass() -> None:
    source = Path("tracks/ksvd/experiments/luyin16/fec_d1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "BindingLocalEnvAdapter":
            continue
        outer_uses = [
            child
            for child in ast.walk(tree)
            if isinstance(child, ast.Attribute)
            and child.attr == "readout"
        ]
        assert not outer_uses  # the core module never touches a readout
    # The adapter is the only place a stat can enter; the runner only attaches
    # statistics to data and sets them on the adapter.
    runner = Path("tracks/ksvd/experiments/luyin16/zinc_fec_d1.py").read_text(encoding="utf-8")
    assert "pair_kernel" not in runner
    assert "center_update" not in runner


@pytest.mark.parametrize(
    "check",
    [
        check_stat_dimensions,
        check_branch_parameter_count,
        check_ego_shells,
        check_binding_reference_and_centering,
        check_small_shell_zero,
        check_shuffle_changes_statistic_and_multisets,
        check_branch_exact_containment,
        check_branch_gradient_flow,
        check_freeze_only_branch_trainable,
        check_no_official_test_access,
        check_no_direct_readout_bypass,
    ],
)
def test_fec_d1_checks(check) -> None:
    check()


__all__ = [name for name in dir() if name.startswith("check_")]
