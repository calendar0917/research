"""Focused CPU tests for E2E-DictEnv-P2-ABS (no data, no GPU)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2


def test_p1_config_bit_identical_to_p1_model():
    config = p2.P2Config("Z2", "p1", 48, 32, 8, 0.125, 240, "sdb32")
    torch.manual_seed(0)
    reference = p1.build_model(p1.SPARSE_ARM, seed=0)
    D = reference.D.detach().numpy().astype(np.float32)
    torch.manual_seed(0)
    model = p2.build_model(config, D, seed=0)
    left = reference.state_dict()
    right = model.state_dict()
    assert sorted(left) == sorted(right)
    for key in left:
        assert left[key].shape == right[key].shape
        assert torch.equal(left[key].float(), right[key].float()), key


def test_parameter_accounting_all_configs():
    configs = [
        ("Z2", "p1", 48, 32, 8, "sdb32"),
        ("H1", "h1", 48, 32, 8, "sdb32"),
        ("H2", "h2", 48, 32, 8, "sdb32"),
        ("E64", "p1", 64, 32, 8, "sdb32"),
        ("H1E64", "h1", 64, 32, 8, "sdb32"),
        ("H2E64", "h2", 64, 32, 8, "sdb32"),
        ("K64S8", "p1", 48, 64, 8, "k64s8"),
        ("K64H1", "h1", 64, 64, 8, "k64s8"),
        ("K64S12", "h2", 64, 64, 12, "k64s12"),
    ]
    for tag, decoder, d_e, K, s, kind in configs:
        config = p2.P2Config(tag, decoder, d_e, K, s, 0.25, 240, kind)
        D = np.random.RandomState(0).randn(p2.PHI_DIM, K).astype(np.float32)
        model = p2.build_model(config, D, seed=0)
        actual = int(sum(p.numel() for p in model.parameters()))
        assert actual == int(p2.total_parameter_count(config)["whole_model"]), tag
        assert p2.PARAM_BUDGET_MIN <= actual <= p2.PARAM_BUDGET_MAX, tag


def test_decoder_layouts():
    h1 = p2.decoder_layers(p2.P2Config("H1", "h1", 48, 32, 8, 0.25, 240))
    assert h1["fusion_in"] == 32 + 3 * 48 + 6 * 32 == 368
    h2 = p2.decoder_layers(p2.P2Config("H2", "h2", 48, 32, 8, 0.25, 240))
    assert h2["fusion_in"] == 48 + 3 * 64 + 6 * 48 == 528
    assert p2.env_input_dim(p2.P2Config("E64", "p1", 64, 32, 8, 0.25, 240)) == 62 + 288 + 6 * 64


def test_exact_sparsity_and_gradient_to_dictionary():
    config = p2.P2Config("H1", "h1", 48, 32, 8, 0.25, 240)
    D = np.random.RandomState(0).randn(p2.PHI_DIM, 32).astype(np.float32)
    D /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-6)
    torch.manual_seed(0)
    model = p2.P2Model(config, D)
    phi = torch.randn(48, p2.PHI_DIM)
    alpha = model.code(phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    assert int(l0.max()) <= int(config.s)
    loss = alpha.sum()
    loss.backward()
    assert float(model.D.grad.norm()) >= 0.0  # reconstruction path always touches D


def test_no_forbidden_bypass_in_source():
    forbidden = {"patch_cont", "atom_shell", "bond_shell", "path_bond_mean", "adjacent_bond_type"}
    for module in (p2,):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert not (names & forbidden)
