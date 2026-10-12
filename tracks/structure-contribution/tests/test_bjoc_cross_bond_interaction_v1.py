"""Tests for cscl-bjoc-cross-bond-interaction-v1 (CBIC).

Core tests are data-free and run in a fresh clone on CPU: the cross-bond
channel algebra and invariances, side-assignment correctness, builder parity
with the published BJOC, parameter matching, forbidden-module / discipline
source scans.  Result-dependent checks skip when the gitignored result JSONs
are absent.

Official valid / official test are never referenced or loaded.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_cross_bond_interaction_v1 as cbic  # noqa: E402
import ijcm_synthetic_v1 as syn  # noqa: E402
import run_bjoc_cross_bond_interaction_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bjoc_cross_bond_interaction_v1.yaml"
RESULTS = ROOT / "results" / "bjoc_cross_bond_interaction_v1"


# ---------------------------------------------------------------------------
# fixtures (no ZINC cache)
# ---------------------------------------------------------------------------


def _tiny_objects(seed: int = 0, n: int = 6, n_types: int = 3):
    rng = np.random.default_rng(seed)
    objs = []
    for _ in range(n):
        n_atoms = int(rng.integers(6, 12))
        atom_types, edges = syn.random_chain_or_tree(rng, n_atoms, n_types, 2)
        objs.append(syn.objects_from_graph(atom_types, edges))
    return objs


@pytest.fixture(scope="module")
def tiny_model():
    mean = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    scale = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    models = {
        arm: cbic.build_arm(arm, seed=0, global_mean=mean, global_scale=scale, device="cpu")
        for arm in cbic.ARMS
    }
    return models, mean, scale


def _flat_side_arrays(objs):
    bond_index, junc_index, junc_bond = [], [], []
    b_off = a_off = 0
    for obj in objs:
        nb = int(obj.bond_index.shape[0])
        bond_index.append(obj.bond_index.astype(np.int64) + a_off)
        junc_index.append(obj.junc_index.astype(np.int64) + a_off)
        junc_bond.append(obj.junc_bond.astype(np.int64) + b_off)
        b_off += nb
        a_off += obj.n_atoms
    return (
        np.concatenate(bond_index, 0),
        np.concatenate(junc_index, 0),
        np.concatenate(junc_bond, 0),
    )


# ---------------------------------------------------------------------------
# cross-bond channel: definition, identity, invariances
# ---------------------------------------------------------------------------


def test_interaction_scale_is_frozen_sqrt96():
    assert cbic.INTERACTION_SCALE == float(np.sqrt(96.0))


def test_cross_bond_q_matches_frozen_formula(tiny_model):
    models, _, _ = tiny_model
    model = models["P"]
    objs = _tiny_objects(seed=1)
    batch = sro.batch_objects(objs)
    with torch.no_grad():
        a = model.encode_atoms(batch)
        b = model.encode_bonds(a, batch)
        w = model.encode_junctions(a, b, batch)
        q = model.cross_bond_q(w, batch)
        s_u, s_v, n_u, n_v = model.side_statistics(w, batch)
    mu_u = s_u / n_u.clamp(min=1.0).unsqueeze(1)
    mu_v = s_v / n_v.clamp(min=1.0).unsqueeze(1)
    expected = cbic.INTERACTION_SCALE * (mu_u * mu_v)
    valid = ((n_u > 0) & (n_v > 0)).unsqueeze(1)
    expected = torch.where(valid, expected, torch.zeros_like(expected))
    assert torch.equal(q, expected)
    # no interaction for a junction-free side
    assert bool((q[~valid.squeeze(1)] == 0).all())


def test_exact_algebraic_decomposition_identity():
    """q / sqrt(96) == (W^2 - V)/2/(n_u n_v) - P_net, with P_net the
    same-side pair correction, exactly (float64)."""
    rng = np.random.default_rng(3)
    objs = _tiny_objects(seed=2, n=10)
    bond_index, junc_index, junc_bond = _flat_side_arrays(objs)
    n_bonds = bond_index.shape[0]
    junc_ids, bond_ids, side = drv.side_flat_arrays(bond_index, junc_index, junc_bond)
    D = 96
    w = rng.normal(size=(junc_index.shape[0], D))
    s_u = np.zeros((n_bonds, D))
    s_v = np.zeros((n_bonds, D))
    Q_u = np.zeros((n_bonds, D))
    Q_v = np.zeros((n_bonds, D))
    n_u = np.bincount(bond_ids[side == 0], minlength=n_bonds)
    n_v = np.bincount(bond_ids[side == 1], minlength=n_bonds)
    np.add.at(s_u, bond_ids[side == 0], w[junc_ids[side == 0]])
    np.add.at(s_v, bond_ids[side == 1], w[junc_ids[side == 1]])
    np.add.at(Q_u, bond_ids[side == 0], w[junc_ids[side == 0]] ** 2)
    np.add.at(Q_v, bond_ids[side == 1], w[junc_ids[side == 1]] ** 2)
    denom = np.maximum(n_u * n_v, 1)[:, None].astype(np.float64)
    X = s_u * s_v / denom
    P_net = 0.5 * ((s_u**2 - Q_u) + (s_v**2 - Q_v)) / denom
    W = s_u + s_v
    V = Q_u + Q_v
    old = (W * W - V) / 2.0 / denom
    assert np.abs((X + P_net) - old).max() < 1e-12
    valid = (n_u > 0) & (n_v > 0)
    assert np.isfinite(X[valid]).all() and np.isfinite(P_net[valid]).all()


def test_each_junction_contributes_once_per_touched_bond():
    objs = _tiny_objects(seed=4, n=8)
    bond_index, junc_index, junc_bond = _flat_side_arrays(objs)
    junc_ids, bond_ids, side = drv.side_flat_arrays(bond_index, junc_index, junc_bond)
    n_bonds = bond_index.shape[0]
    # every incidence row is unique
    rows = np.stack([junc_ids, bond_ids, side], 1)
    assert np.unique(rows, axis=0).shape[0] == rows.shape[0]
    # each junction appears exactly twice (its two bond ends)
    counts = np.bincount(junc_ids, minlength=junc_index.shape[0])
    assert int(counts.sum()) == 2 * junc_index.shape[0]
    assert (counts == 2).all()  # exactly the two bonds of each junction centre
    # every incidence row is accounted for by the per-bond side counts
    n_u = np.bincount(bond_ids[side == 0], minlength=n_bonds)
    n_v = np.bincount(bond_ids[side == 1], minlength=n_bonds)
    assert int(n_u.sum() + n_v.sum()) == 2 * junc_index.shape[0]


def test_endpoint_swap_q_invariant(tiny_model):
    models, _, _ = tiny_model
    model = models["P"]
    atom_types = [0, 1, 2, 0, 1]
    edges = [(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 4, 0)]
    o1 = syn.objects_from_graph(atom_types, edges)
    o2 = syn.objects_from_graph(atom_types, [(v, u, t) for u, v, t in edges])
    with torch.no_grad():
        f1 = model.forward_features(sro.batch_objects([o1]))
        f2 = model.forward_features(sro.batch_objects([o2]))
    assert torch.equal(f1["q"], f2["q"])


def test_junction_endpoint_swap_q_invariant(tiny_model):
    models, _, _ = tiny_model
    model = models["P"]
    objs = _tiny_objects(seed=6, n=4)
    batch = sro.batch_objects(objs)
    with torch.no_grad():
        f1 = model.forward_features(batch)
        swapped = dict(batch)
        ji = batch["junc_index"].clone()
        ji[:, 0], ji[:, 2] = batch["junc_index"][:, 2].clone(), batch["junc_index"][:, 0].clone()
        swapped["junc_index"] = ji
        f2 = model.forward_features(swapped)
    assert torch.equal(f1["q"], f2["q"])


def test_atom_relabel_q_invariant(tiny_model):
    models, _, _ = tiny_model
    model = models["P"]
    objs = _tiny_objects(seed=7, n=3)
    obj = objs[0]
    perm = np.arange(obj.n_atoms - 1, -1, -1)
    cat = np.argmax(obj.atom_feat[:, : sro.ATOM_CATEGORIES], axis=1).astype(np.int64)
    x = cat[perm].reshape(-1, 1)
    bonds = np.asarray(obj.bond_index, dtype=np.int64)
    directed = np.concatenate([perm[bonds], perm[bonds[:, ::-1]]], axis=0)
    relabelled = sro.build_mol_objects(-1, x, directed.T, np.concatenate([obj.bond_type, obj.bond_type]))
    with torch.no_grad():
        q1 = model.forward_features(sro.batch_objects([obj]))["q"]
        q2 = model.forward_features(sro.batch_objects([relabelled]))["q"]
    # relabelling permutes the canonical bond order; compare the bond-indexed
    # q rows as a multiset (lexicographic row sort)
    def _sorted(t):
        arr = t.detach().cpu().numpy()
        order = np.lexsort(arr[:, ::-1].T)
        return arr[order]

    assert _sorted(q1).shape == _sorted(q2).shape
    assert np.allclose(_sorted(q1), _sorted(q2), atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# builder parity, parameter matching, forbidden modules
# ---------------------------------------------------------------------------


def test_all_arms_have_428865_params_and_equal_init(tiny_model):
    models, mean, scale = tiny_model
    counts = {arm: sum(p.numel() for p in m.parameters()) for arm, m in models.items()}
    assert counts == {"N": 428865, "O": 428865, "P": 428865}
    hashes = {arm: cbic.state_dict_hash(m) for arm, m in models.items()}
    assert len(set(hashes.values())) == 1
    # O and P must produce identical predictions at init (q enters through
    # zero-initialised composer columns only)
    objs = _tiny_objects(seed=8, n=4)
    batch = sro.batch_objects(objs)
    with torch.no_grad():
        p_o = models["O"](batch)
        p_p = models["P"](batch)
    assert torch.equal(p_o, p_p)


def test_build_cbic_copies_published_weights_and_zero_inits_new_columns(tiny_model):
    models, mean, scale = tiny_model
    model = models["P"]
    torch.manual_seed(0)
    base = src.BJOC(src.BJOCConfig(), mean, scale, src.ORGANIZED)
    base_sd = base.state_dict()
    for key, target in model.state_dict().items():
        if key == "group_composer.lins.0.weight":
            assert torch.equal(target[:, : base_sd[key].shape[1]], base_sd[key])
            assert bool((target[:, base_sd[key].shape[1] :] == 0).all())
        else:
            assert torch.equal(target, base_sd[key]), key
    # inventory arithmetic: exactly 96 * compose_hidden new parameters
    assert (428865 - 413505) == 96 * 160


def test_forbidden_modules_absent(tiny_model):
    models, _, _ = tiny_model
    names = {type(m).__name__ for m in models["P"].modules()}
    assert names <= {"CBIC", "BJOC", "_MLP", "Linear", "ModuleList"}
    assert cbic.forbidden_module_scan(models["P"]) == []


def test_parameter_inventory_exact(tiny_model):
    models, _, _ = tiny_model
    inv = cbic.parameter_inventory(models["P"])
    assert inv["parameter_exact"]
    assert inv["base_total"] == 413505
    assert inv["interaction_parameters_added"] == 96 * 160


# ---------------------------------------------------------------------------
# protocol / discipline source scans
# ---------------------------------------------------------------------------


def test_config_exists_and_declares_protocol():
    assert CONFIG.exists()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == "cscl-bjoc-cross-bond-interaction-v1"
    assert cfg["official_splits"] == "never_read"
    assert cfg["budget"]["gpu_hours_max"] <= 2.5
    assert cfg["budget"]["concurrency"] == 1
    bans = " ".join(cfg["bans"])
    for token in ("official valid", "official test", "GNN", "attention", "bilinear", "rank search"):
        assert token in bans


def _sources():
    return {
        "driver": (drv.__file__ and Path(drv.__file__).read_text(encoding="utf-8")) or "",
        "model": (_CODE_DIR / "bjoc_cross_bond_interaction_v1.py").read_text(encoding="utf-8"),
    }


def test_no_official_valid_or_test_references():
    for name, text in _sources().items():
        for token in ("encoded_valid", "encoded_test", "official_valid.pt", "official_test.pt", "load_official_valid", "load_official_test"):
            assert token not in text, f"{name} references {token}"
        if name == "driver":
            assert '"official_valid_or_test_loaded": False' in text


def test_no_component_supervision_tokens():
    for name, text in _sources().items():
        for token in ("target_decomposition", "g_target", "ell_target", "aux_loss", "teacher"):
            assert token not in text, f"{name} references {token}"


def test_predict_and_decide_do_not_read_labels():
    driver = _sources()["driver"]
    tree = ast.parse(driver)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("cmd_predict", "cmd_params", "cmd_gates", "cmd_monitor_gate"):
            body = ast.get_source_segment(driver, node) or ""
            assert "load_y_rows" not in body or node.name == "cmd_monitor_gate", node.name


def test_cross_bond_forward_source_is_the_only_difference():
    """The CBIC forward must extend the published BJOC compose input by q only."""
    model_src = _sources()["model"]
    tree = ast.parse(model_src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CBIC")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {"bond_side_assignment", "side_statistics", "cross_bond_q", "forward_features"} <= methods
    # q is appended to the composer input exactly once
    seg = model_src[model_src.index("compose_in = torch.cat") :]
    assert seg.count("q") >= 1


def test_no_bilinear_or_second_channel():
    model_src = _sources()["model"]
    for token in ("nn.Bilinear", "einsum", "matmul(w", "rank_search", "second_interaction", "interaction2", "bilinear_layer"):
        assert token not in model_src, token


# ---------------------------------------------------------------------------
# result-dependent checks (skip in a fresh clone)
# ---------------------------------------------------------------------------


def test_phase_a_gate_artifact_if_present():
    path = RESULTS / "phase_a_gate.json"
    if not path.exists():
        pytest.skip("phase-a gate artifact not present in this clone")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"] == "PHASE_A_PURCHASE_GRANTED"
    assert all(payload["checks"].values())
    # the disclosed non-gating linear probe number is reported
    assert "linear_view_probe_residual" in payload["reported_not_gating"]


def test_correctness_gates_if_present():
    path = RESULTS / "correctness_gates.json"
    if not path.exists():
        pytest.skip("correctness gates not present in this clone")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["all_gates_ok"]
    assert payload["gate15_parameters_equal"]["params"] == {"N": 428865, "O": 428865, "P": 428865}
    assert payload["official_valid_or_test_loaded"] is False


def test_freezing_manifest_if_present():
    path = RESULTS / "protocol_freeze_manifest.json"
    if not path.exists():
        pytest.skip("freeze manifest not present in this clone")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol"] == "cscl-bjoc-cross-bond-interaction-v1"
    assert "code_sha256" in payload and "artifacts" in payload
