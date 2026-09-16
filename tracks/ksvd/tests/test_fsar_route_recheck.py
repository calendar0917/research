"""FSAR route-recheck correctness tests.

Covers exactly the two pre-registered objects
(``notes/fsar_route_recheck_preregistration.md``):

* the operator-matched assignment-independent null ``B_indep`` -- same modules,
  same active parameters, same multiplicative machinery, no pairing;
* the explicit local structural basis with the *FSAR-v1* ``A``.

Plus the inherited contract checks (bit-for-bit FSAR-v1 latent equivalence, no
mixed bypass, relabel invariance, gradient viability, parameter accounting).
Official ZINC test is never touched.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import fsar
from tracks.ksvd.experiments.luyin16 import fsar_route_recheck as rr
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)
from tracks.ksvd.tests import test_fsar_v2 as fx


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _encode_one(data):
    return fx._encode_one(data)


def _batch(data_list):
    return rr.fsar_route_collate(data_list)


def _fresh_model(mode: str, seed: int = 0) -> rr.PatchPathFSARRouteModel:
    torch.manual_seed(int(seed))
    model = rr.PatchPathFSARRouteModel(mode=mode)
    model.eval()
    return model


def _relabel(molecule, permutation):
    inverse = [0] * len(permutation)
    for new, old in enumerate(permutation):
        inverse[old] = new
    relabeled_edges = []
    relabeled_bonds = []
    for (left, right), bond in zip(
        molecule.edge_index.t().tolist(), molecule.edge_attr.view(-1).tolist()
    ):
        if int(left) >= int(right):
            continue
        relabeled_edges.append(tuple(sorted((inverse[int(left)], inverse[int(right)]))))
        relabeled_bonds.append(int(bond))
    order = sorted(range(len(relabeled_edges)), key=lambda i: relabeled_edges[i])
    return fx._molecule(
        [molecule.x.view(-1)[old].item() for old in permutation],
        [relabeled_edges[i] for i in order],
        [relabeled_bonds[i] for i in order],
        y=float(molecule.y.view(-1)[0]),
    )


def _grad_map(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
    }


# ---------------------------------------------------------------------------
# bit-for-bit FSAR-v1 equivalence for the latent aligned route
# ---------------------------------------------------------------------------


def test_latent_align_matches_fsar_v1() -> None:
    for mode in ("SA", "SAB"):
        torch.manual_seed(3)
        reference = fsar.PatchPathFSARModel(mode=mode)
        torch.manual_seed(3)
        route = rr.PatchPathFSARRouteModel(mode=mode)
        reference.eval()
        route.eval()
        reference_state = reference.state_dict()
        route_state = route.state_dict()
        assert set(reference_state) == set(route_state), mode
        for key in reference_state:
            assert reference_state[key].shape == route_state[key].shape, (mode, key)
            assert torch.equal(reference_state[key], route_state[key]), (mode, key)
        batch = _batch([_encode_one(fx._asymmetric_molecule())])
        with torch.no_grad():
            assert torch.equal(reference(batch), route(batch)), mode
    # SAM capacity control must also be identical.
    torch.manual_seed(5)
    reference_sam = fsar.PatchPathFSARModel(mode="SAM")
    torch.manual_seed(5)
    route_sam = rr.PatchPathFSARRouteModel(mode="SAM")
    for key, value in reference_sam.state_dict().items():
        assert torch.equal(value, route_sam.state_dict()[key]), key


# ---------------------------------------------------------------------------
# operator / active-parameter identity of the aligned vs independent null
# ---------------------------------------------------------------------------


def _assert_parameter_identity(align_mode: str, null_mode: str) -> None:
    align = _fresh_model(align_mode, seed=0)
    null = _fresh_model(null_mode, seed=0)
    align_state = align.state_dict()
    null_state = null.state_dict()
    assert set(align_state) == set(null_state), (align_mode, null_mode)
    for key in align_state:
        assert align_state[key].shape == null_state[key].shape, (align_mode, key)
    assert sum(p.numel() for p in align.parameters()) == sum(
        p.numel() for p in null.parameters()
    )
    align_breakdown = rr.parameter_breakdown(align)
    null_breakdown = rr.parameter_breakdown(null)
    assert align_breakdown["B_params"] == null_breakdown["B_params"]
    assert align_breakdown["A_params"] == null_breakdown["A_params"]
    assert align_breakdown["local_S_params"] == null_breakdown["local_S_params"]
    assert align_breakdown["relation_core_params"] == null_breakdown["relation_core_params"]
    # Same *active* prediction-path budget (pre-registered section 10).
    batch = _batch(
        [_encode_one(fx._asymmetric_molecule()), _encode_one(fx._asymmetric_molecule_two())]
    )
    _backward(align, batch)
    _backward(null, batch)
    align_active = rr.active_path_from_grads(align)
    null_active = rr.active_path_from_grads(null)
    assert align_active["all_modules_active"], align_mode
    assert null_active["all_modules_active"], null_mode
    assert (
        align_active["active_prediction_path_params"]
        == null_active["active_prediction_path_params"]
    ), (align_active["active_prediction_path_params"], null_active["active_prediction_path_params"])
    assert align_active["active_prediction_path_params"] == align_breakdown["total"]
    assert null_active["active_prediction_path_params"] == null_breakdown["total"]


def test_latent_aligned_null_parameter_identity() -> None:
    _assert_parameter_identity("SAB", "SABI")


def test_explicit_aligned_null_parameter_identity() -> None:
    _assert_parameter_identity("SABE", "SABEI")


# ---------------------------------------------------------------------------
# B_indep correctness
# ---------------------------------------------------------------------------


def test_b_indep_same_marginal_invariance() -> None:
    original = _encode_one(fx._asymmetric_molecule())
    permuted = fx._permute_context_attributes(original, change_marginal=False)
    for mode in ("SABI", "SABEI"):
        model = _fresh_model(mode, seed=1)
        with torch.no_grad():
            channels_a = model.encoder.forward_channels(_batch([original]))
            channels_b = model.encoder.forward_channels(_batch([permuted]))
        assert torch.allclose(
            channels_a["attributes"],
            channels_b["attributes"],
            atol=1e-6,
            rtol=0.0,
        ), mode
        assert torch.allclose(
            channels_a["structure"], channels_b["structure"], atol=1e-6, rtol=0.0
        ), mode
        delta = float(
            (channels_a["binding"] - channels_b["binding"]).abs().max()
        )
        assert delta <= 1e-6, (
            f"{mode}: assignment-independent null must be invariant to the "
            f"attribute<->role assignment, max |delta| = {delta}"
        )


def test_b_align_assignment_sensitivity() -> None:
    original = _encode_one(fx._asymmetric_molecule())
    permuted = fx._permute_context_attributes(original, change_marginal=False)
    for mode in ("SAB", "SABE"):
        model = _fresh_model(mode, seed=1)
        with torch.no_grad():
            channels_a = model.encoder.forward_channels(_batch([original]))
            channels_b = model.encoder.forward_channels(_batch([permuted]))
        assert torch.equal(
            channels_a["structure"], channels_b["structure"]
        ), mode
        delta = float((channels_a["binding"] - channels_b["binding"]).norm())
        assert delta > 1e-5, f"{mode}: aligned binding must react to assignment"


# ---------------------------------------------------------------------------
# shared chemistry tensor / shared structural basis tensor
# ---------------------------------------------------------------------------


def test_chemistry_coverage_shared() -> None:
    """Both align and null read the same learned atom/bond entity tensors."""
    batch = _batch([_encode_one(fx._asymmetric_molecule())])
    for mode in ("SAB", "SABI", "SABE", "SABEI"):
        model = _fresh_model(mode, seed=2)
        with torch.no_grad():
            channels = model.encoder.forward_channels(batch)
            expected_node = model.encoder.atom_mlp(
                model.encoder.atom_embedding(batch.struct_atom.long())
            )
            expected_edge = model.encoder.bond_mlp(
                model.encoder.bond_embedding(batch.struct_bond.long())
            )
        assert torch.equal(channels["node_attribute"], expected_node), mode
        assert torch.equal(channels["edge_attribute"], expected_edge), mode


def test_structure_coverage_shared() -> None:
    """Both align and null read the same structural side (role or basis)."""
    # explicit: perturbing the basis must move S *and* B in align and null
    for mode in ("SABE", "SABEI"):
        encoded = _encode_one(fx._asymmetric_molecule())
        perturbed = fx._clone(encoded)
        perturbed.node_basis = perturbed.node_basis * 1.3 + 0.1
        model = _fresh_model(mode, seed=2)
        with torch.no_grad():
            a = model.encoder.forward_channels(_batch([encoded]))
            b = model.encoder.forward_channels(_batch([perturbed]))
        assert float((a["structure"] - b["structure"]).abs().max()) > 1e-5, mode
        assert float((a["binding"] - b["binding"]).abs().max()) > 1e-5, mode
    # latent: perturbing the rooted shell must move S *and* B in align and null
    for mode in ("SAB", "SABI"):
        encoded = _encode_one(fx._asymmetric_molecule())
        perturbed = fx._clone(encoded)
        perturbed.struct_dist = (
            perturbed.struct_dist + 1
        ) % rr.v1.N_DISTANCE_BINS
        model = _fresh_model(mode, seed=2)
        with torch.no_grad():
            a = model.encoder.forward_channels(_batch([encoded]))
            b = model.encoder.forward_channels(_batch([perturbed]))
        assert float((a["structure"] - b["structure"]).abs().max()) > 1e-5, mode
        assert float((a["binding"] - b["binding"]).abs().max()) > 1e-5, mode


def test_relabel_invariance() -> None:
    molecule = fx._asymmetric_molecule()
    permutation = [3, 0, 5, 1, 4, 2]
    relabeled = _relabel(molecule, permutation)
    for mode in ("SAB", "SABI", "SABE", "SABEI"):
        model = _fresh_model(mode, seed=4)
        with torch.no_grad():
            prediction_a = model(_batch([_encode_one(molecule)]))
            prediction_b = model(_batch([_encode_one(relabeled)]))
        assert torch.allclose(prediction_a, prediction_b, atol=1e-5, rtol=0.0), mode


# ---------------------------------------------------------------------------
# explicit basis
# ---------------------------------------------------------------------------


def test_explicit_basis_chemistry_purity() -> None:
    original = fx._asymmetric_molecule()
    perturbed = fx._asymmetric_molecule()
    perturbed.x = (perturbed.x + 5) % v2.ATOM_CATEGORIES
    perturbed.edge_attr = (perturbed.edge_attr + 1) % v2.BOND_CATEGORIES
    original_enc = _encode_one(original)
    perturbed_enc = _encode_one(perturbed)
    assert torch.equal(original_enc.node_basis, perturbed_enc.node_basis)
    assert torch.equal(original_enc.edge_basis, perturbed_enc.edge_basis)
    for mode in ("SAE", "SABE", "SABEI"):
        model = _fresh_model(mode, seed=6)
        with torch.no_grad():
            s_a = model.encoder.forward_channels(_batch([original_enc]))["structure"]
            s_b = model.encoder.forward_channels(_batch([perturbed_enc]))["structure"]
        assert torch.equal(s_a, s_b), f"{mode}: explicit S must be chemistry-free"


def test_explicit_basis_relabel_equivariance() -> None:
    molecule = fx._asymmetric_molecule()
    relabeled = _relabel(molecule, [3, 0, 5, 1, 4, 2])
    original_enc = _encode_one(molecule)
    relabeled_enc = _encode_one(relabeled)
    original_rows = np.sort(original_enc.node_basis.numpy(), axis=0)
    relabeled_rows = np.sort(relabeled_enc.node_basis.numpy(), axis=0)
    assert np.allclose(original_rows, relabeled_rows, atol=1e-6)
    original_edges = np.sort(original_enc.edge_basis.numpy(), axis=0)
    relabeled_edges = np.sort(relabeled_enc.edge_basis.numpy(), axis=0)
    assert np.allclose(original_edges, relabeled_edges, atol=1e-6)


def test_explicit_path_has_no_message_passing() -> None:
    for mode in ("SAE", "SABE", "SABEI"):
        encoder = _fresh_model(mode, seed=6).encoder
        assert encoder.message is None and encoder.update is None, mode
        assert encoder.root_embedding is None, mode
        assert encoder.distance_embedding is None, mode
        assert getattr(encoder, "topology_base", None) is None, mode
    source = pathlib.Path(rr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"struct_src", "struct_dst", "message", "update"}
    found_any = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_explicit_structure",
            "_explicit_binding",
        ):
            found_any = True
            used = {
                child.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            }
            assert not (forbidden & used), (
                f"{node.name} must not touch adjacency / message passing: "
                f"{sorted(forbidden & used)}"
            )
    assert found_any, "explicit structural routines not found"


# ---------------------------------------------------------------------------
# inherited contract checks
# ---------------------------------------------------------------------------


def test_no_mixed_bypass() -> None:
    for mode in rr.ALL_MODES:
        model = rr.PatchPathFSARRouteModel(mode=mode)
        assert not hasattr(model, "attribute_encoder")
        assert not hasattr(model, "structural_encoder")
        assert not hasattr(model, "typed_embedding")
        assert not hasattr(model.encoder, "a_exact_mlp")
    data = _encode_one(fx._asymmetric_molecule())
    for attribute in (
        "patch_cont",
        "patch_context",
        "parent_token",
        "global_context",
        "typed_token",
    ):
        assert not hasattr(data, attribute)
    # the raw exact marginal interface of FSAR-v2 is deliberately absent
    source = pathlib.Path(rr.__file__).read_text(encoding="utf-8")
    for token in ("a_exact_mlp", "raw_marginal_vector", "A_RAW_DIM"):
        assert token not in source


def _backward(model, batch, optimizer_steps: int = 2):
    """Two real optimizer steps then a fresh backward.

    ``center_update``'s last layer is zero-initialised, so the pair/centre stack
    only receives a task gradient from the second step on.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = None
    for _ in range(int(optimizer_steps)):
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
    loss.backward()
    return loss


def test_gradient_viability_all_modes() -> None:
    batch = _batch([_encode_one(fx._asymmetric_molecule())])
    for mode in rr.ALL_MODES:
        model = rr.PatchPathFSARRouteModel(mode=mode)
        _backward(model, batch)
        payload = rr.active_path_from_grads(model)
        assert payload["all_modules_active"], mode
        assert payload["active_prediction_path_params"] > 0, mode
        checkpoints = {
            "atom_embedding": model.encoder.atom_embedding.weight,
            "atom_mlp": model.encoder.atom_mlp[0].weight,
            "bond_embedding": model.encoder.bond_embedding.weight,
            "bond_mlp": model.encoder.bond_mlp[0].weight,
            "node_init": model.node_init[0].weight,
            "relation_encoder": model.relation_encoder[0].weight,
            "pair_encoder": model.pair_encoder[0].weight,
            "center_update": model.center_update[0].weight,
            "head": model.head[-1].weight,
        }
        if rr.MODE_LOCAL_S[mode] is not None:
            checkpoints["structure_pool"] = model.encoder.structure_pool[0].weight
        if rr.MODE_BINDING[mode] is not None:
            for name in (
                "node_role_projection",
                "node_attribute_projection",
                "edge_role_projection",
                "edge_attribute_projection",
            ):
                checkpoints[name] = getattr(model.encoder, name).weight
        if rr.MODE_CAPACITY.get(mode, False):
            checkpoints["capacity_mlp"] = model.capacity_mlp[0].weight
        for name, parameter in checkpoints.items():
            grad = parameter.grad
            assert grad is not None and bool(torch.isfinite(grad).all()), (
                f"{mode}:{name} gradient missing/non-finite"
            )
            assert float(grad.abs().sum()) > 0.0, f"{mode}:{name} zero gradient"


def test_b_indep_module_gradients_alive() -> None:
    """Every aligned-null B module must be in the prediction path."""
    two = fx._asymmetric_molecule_two()
    batch = _batch([_encode_one(fx._asymmetric_molecule()), _encode_one(two)])
    for mode in ("SABI", "SABEI"):
        model = rr.PatchPathFSARRouteModel(mode=mode)
        _backward(model, batch)
        encoder = model.encoder
        for name in (
            "node_role_projection",
            "node_attribute_projection",
            "edge_role_mlp",
            "edge_attribute_mlp",
            "edge_role_projection",
            "edge_attribute_projection",
            "binding_fuse",
        ):
            module = getattr(encoder, name)
            total = sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            assert total > 0.0, f"{mode}:{name} dead in the independent null"
        # the analytical first-moment block of binding_fuse is exactly inactive
        weight_grad = encoder.binding_fuse[0].weight.grad
        assert float(weight_grad[:, : rr.B_DIM].abs().sum()) == 0.0, mode
        assert (
            float(
                weight_grad[
                    :, 2 * rr.B_DIM : 3 * rr.B_DIM
                ].abs().sum()
            )
            == 0.0
        ), mode
        assert float(weight_grad[:, rr.B_DIM : 2 * rr.B_DIM].abs().sum()) > 0.0
        assert float(weight_grad[:, 3 * rr.B_DIM :].abs().sum()) > 0.0


def test_parameter_accounting() -> None:
    breakdowns = {}
    for mode in rr.ALL_MODES:
        breakdowns[mode] = rr.parameter_breakdown(
            rr.PatchPathFSARRouteModel(mode=mode)
        )
    totals = {mode: row["total"] for mode, row in breakdowns.items()}
    assert totals["A"] < totals["SA"] < totals["SAB"]
    assert totals["SAE"] < totals["SABE"]
    assert totals["SAB"] == totals["SABI"]
    assert totals["SABE"] == totals["SABEI"]
    assert breakdowns["SAB"]["B_params"] == breakdowns["SABI"]["B_params"]
    assert breakdowns["SABE"]["B_params"] == breakdowns["SABEI"]["B_params"]
    assert breakdowns["SAB"]["local_S_params"] == breakdowns["SABI"]["local_S_params"]
    for mode, row in breakdowns.items():
        assert row["total"] <= 200_000, mode
        assert row["dataset_dependent_vocabulary_params"] == 0
        assert row["trainable"] == row["total"]
    # the null has no extra MLP: the B group is the same module set
    for align, null in (("SAB", "SABI"), ("SABE", "SABEI")):
        assert breakdowns[align]["B_params"] > 0
        assert breakdowns[null]["capacity_mlp_params"] == 0
