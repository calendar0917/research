"""cscl-structural-relation-composer-v1 correctness / mechanism-liveness tests.

Fast tests (no ZINC data, no checkpoints) cover the twenty engineering gates
of the protocol at the level of construction, symmetry, accounting,
organisation marginals, arm pairing, gradients, invariance and protocol
guards.  Slow tests (marked ``slow``) load the real official-train PyG rows
and the label-blind object cache.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import cscl_units as cu  # noqa: E402

CFG = src.BJOCConfig()
ZERO_MEAN = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
ONE_SCALE = np.ones(sro.GLOBAL_DIM, dtype=np.float32)


def _chain(n: int, atom_types, bond_types=None):
    bonds = [(i, i + 1, 0) for i in range(n - 1)] if bond_types is None else [(i, i + 1, bond_types[i]) for i in range(n - 1)]
    return _mol(atom_types, bonds)


def _mol(atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    return sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64))


def _batch(objs):
    return sro.batch_objects(objs)


def _model(mode=src.ORGANIZED, cfg=CFG, seed=0):
    return src.build_bjoc(cfg, seed=seed, organization_mode=mode, global_mean=ZERO_MEAN, global_scale=ONE_SCALE)


# ---------------------------------------------------------------------------
# fast: object construction and accounting
# ---------------------------------------------------------------------------


def test_gate2_pyg_double_edges_are_deduplicated_once():
    obj = _chain(5, [0, 1, 0, 1, 0], [0, 1, 0, 1])
    assert obj.bond_index.shape[0] == 4  # five atoms, four physical bonds
    assert len(set(map(tuple, obj.bond_index.tolist()))) == 4


def test_gate5_gate6_gate7_bond_and_junction_accounting():
    obj = _mol([0, 1, 0, 1], [(0, 1, 0), (1, 2, 0), (2, 3, 0)])
    # chain X-Y-X-Y: two junctions (at atoms 1 and 2), each touching 2 bonds
    assert obj.junc_index.shape[0] == 2
    assert obj.junc_bond.shape == (2, 2)
    counts = np.zeros(obj.bond_index.shape[0], dtype=np.int64)
    for b1, b2 in obj.junc_bond:
        counts[b1] += 1
        counts[b2] += 1
    assert counts.sum() == 2 * obj.junc_index.shape[0]
    # every junction has i<k and a real centre distinct from both
    for i, j, k in obj.junc_index:
        assert i < k and j != i and j != k


def test_gate1_atom_relabelling_invariance_of_objects():
    obj = _chain(6, [0, 1, 2, 1, 0, 3], [0, 1, 0, 1, 0])
    perm = np.array([3, 0, 5, 1, 4, 2])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    atom_types = [0, 1, 2, 1, 0, 3]
    bonds = [(i, i + 1, bt) for i, bt in enumerate([0, 1, 0, 1, 0])]
    relabelled = []
    for u, v, bt in bonds:
        a, b = int(inv[u]), int(inv[v])
        relabelled.append((min(a, b), max(a, b), bt))
    obj2 = _mol([atom_types[p] for p in perm], sorted(relabelled))
    # marginals of the object tensors are label-invariant
    assert sorted(map(tuple, obj.atom_feat.tolist())) == sorted(map(tuple, obj2.atom_feat.tolist()))
    assert sorted(map(tuple, obj.junc_feat.tolist())) == sorted(map(tuple, obj2.junc_feat.tolist()))
    assert sorted(map(tuple, obj.bond_feat.tolist())) == sorted(map(tuple, obj2.bond_feat.tolist()))
    assert np.allclose(obj.global_feat, obj2.global_feat)


def test_gate3_gate4_endpoint_swap_invariance():
    obj = _mol([0, 1, 0, 1], [(0, 1, 0), (1, 2, 1), (2, 3, 0)])
    model = _model()
    b = _batch([obj])
    _, feats = model(b, return_features=True)
    # direct endpoint swap: reverse the stored direction of every bond and the
    # (i, k) order of every junction; both encoders are constructed symmetric
    b_swap = dict(b)
    b_swap["bond_index"] = b["bond_index"][:, [1, 0]]
    b_swap["junc_index"] = b["junc_index"][:, [2, 1, 0]]
    a = model.encode_atoms(b_swap)
    b_enc = model.encode_bonds(a, b_swap)
    w = model.encode_junctions(a, b_enc, b_swap)
    assert torch.allclose(b_enc, feats["b"], atol=1e-6)
    assert torch.allclose(w, feats["w"], atol=1e-6)


def test_gate8_gate9_gate10_null_marginals_and_pairing():
    objs = [_mol([0, 1, 0, 1], [(0, 1, 0), (1, 2, 1), (2, 3, 0)]), _mol([1, 0, 1, 2], [(0, 1, 1), (1, 2, 0)])]
    b = _batch(objs)
    m_org = _model(src.ORGANIZED)
    m_null = _model(src.NULL)
    assert src.state_dict_hash(m_org) == src.state_dict_hash(m_null)
    assert src.parameter_audit(CFG, m_org)["parameter_exact"]
    assert sum(p.numel() for p in m_org.parameters()) == sum(p.numel() for p in m_null.parameters())
    with torch.no_grad():
        w = m_org.encode_junctions(m_org.encode_atoms(b), m_org.encode_bonds(m_org.encode_atoms(b), b), b)
        Wr, Vr, nb_r = m_org.group_to_bonds(w, b, src.ORGANIZED)
        Wn, Vn, nb_n = m_null.group_to_bonds(w, b, src.NULL)
    assert torch.equal(nb_r, nb_n)
    assert torch.allclose(Wr.sum(0), Wn.sum(0), atol=1e-3)
    assert torch.allclose(Vr.sum(0), Vn.sum(0), atol=1e-3)


def test_gate11_gradients_reach_all_object_encoders():
    b = _batch([_mol([0, 1, 0, 1, 2], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 0)])])
    model = _model()
    loss = model(b).sum()
    loss.backward()
    groups = {"atom_encoder": False, "bond_encoder": False, "junction_encoder": False, "group_composer": False, "reader": False}
    for name, p in model.named_parameters():
        if p.grad is not None and float(p.grad.abs().sum()) > 0:
            for key in groups:
                if name.startswith(key):
                    groups[key] = True
    assert all(groups.values()), groups
    # the null arm also trains its junction encoder (mean depends on w)
    model_n = _model(src.NULL)
    model_n(b).sum().backward()
    assert float(model_n.junction_encoder.lins[0].weight.grad.abs().sum()) > 0


def test_gate12_forward_is_y_independent():
    b = _batch([_mol([0, 1, 0, 1], [(0, 1, 0), (1, 2, 1), (2, 3, 0)])])
    model = _model()
    p1 = model(b)
    b2 = dict(b)
    b2["y"] = torch.randn(1) * 100
    p2 = model(b2)
    assert torch.equal(p1, p2)


def test_gate13_batch_grouping_invariance():
    objs = [_mol([0, 1, 0, 1, 2], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 0)]), _mol([0, 1, 0, 1], [(0, 1, 1), (1, 2, 0), (2, 3, 0)])]
    model = _model()
    full = model(_batch(objs))
    single = torch.cat([model(_batch([o])) for o in objs])
    assert torch.allclose(full, single, atol=1e-6)


def test_gate14_save_load_replay(tmp_path: Path):
    model = _model()
    b = _batch([_mol([0, 1, 0, 1, 2], [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 0)])])
    p1 = model(b)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    model2 = _model()
    model2.load_state_dict(torch.load(path, weights_only=False))
    p2 = model2(b)
    assert torch.equal(p1, p2)


def test_gate15_no_nan_inf_and_robust_to_empty_junctions():
    objs = [_mol([0, 1], [(0, 1, 0)])]  # one bond, zero junctions
    b = _batch(objs)
    # sro rejects single-atom/bond-only molecules? it builds them fine
    model = _model()
    p, feats = model(b, return_features=True)
    assert torch.isfinite(p).all()
    assert torch.isfinite(feats["W"]).all()
    assert float(feats["n_b"].sum()) == 0.0
    assert torch.isfinite(_model(src.NULL)(b)).all()


def test_gate16_no_forbidden_architecture():
    assert src.forbidden_module_scan(_model()) == []
    assert src.forbidden_module_scan(_model(src.NULL)) == []


def test_object_level_marginal_preserving_ownership_switch():
    """The explicit Phase-D witness: identical marginals, different ownership."""
    a, b = src.counterexample_object_pair()
    assert src.object_marginal_signature(a) == src.object_marginal_signature(b)
    assert src.per_bond_junction_tokens(a) != src.per_bond_junction_tokens(b)
    batch = src.batch_object_instances([a, b])
    model = _model()
    with torch.no_grad():
        fo = model.forward_features(batch, organization_mode=src.ORGANIZED)
        fn = model.forward_features(batch, organization_mode=src.NULL)
    assert float((fo["W"][:4] - fo["W"][4:]).abs().max()) > 1e-6
    assert float((fn["W"][:4] - fn["W"][4:]).abs().max()) == 0.0
    # incidence target actually depends on the ownership mapping
    assert a.target_incidence != b.target_incidence
    assert a.target_marginal == b.target_marginal
    # A: (2-0)^2*1 + (0-2)^2*1 + 0*2 + (1*1)*1 + (1*1)*1 = 4+4+1+1 = 10
    assert a.target_incidence == 10.0
    # B: every bond sees {X,Y}: (0)^2 + mixed-count bonus only for token-1 bonds = 2
    assert b.target_incidence == 2.0


def test_object_batch_shapes_and_targets():
    rng = np.random.default_rng(0)
    insts = [src.random_object_instance(rng, 4, 5) for _ in range(3)]
    batch = src.batch_object_instances(insts)
    assert batch["bond_feat"].shape[0] == 12
    assert batch["junc_bond"].shape == (15, 2)
    assert batch["target_incidence"].shape == (3,)
    assert torch.isfinite(batch["bond_feat"]).all()


def test_gate18_gate19_locked_dev_not_used_in_training_code():
    driver = (_CODE_DIR / "run_structural_relation_composer_v1.py").read_text(encoding="utf-8")
    # locked labels are read only inside the eval stage
    assert 'locked_labels_read' in driver
    assert 'y_locked' in driver
    # training functions must not touch the locked key
    train_bjoc = driver[driver.index("def train_bjoc_arm") : driver.index("def predict_bjoc")]
    assert '"locked"' not in train_bjoc
    train_full = driver[driver.index("def train_full_y_matched") : driver.index("def cmd_train_f")]
    assert '"locked"' not in train_full
    # official valid / test never appear as loaded objects
    assert 'load_official_valid' not in driver
    assert 'load_official_test' not in driver


def test_gate20_official_test_never_touched():
    for name in ("structural_relation_objects_v1.py", "structural_relation_composer_v1.py", "run_structural_relation_composer_v1.py"):
        text = (_CODE_DIR / name).read_text(encoding="utf-8")
        assert "official_test" not in text or "never" in text or "official_test_loaded" in text
        assert "load_test" not in text
        assert "split=\"test\"" not in text and "split='test'" not in text


def test_frozen_split_is_deterministic_and_grouped():
    sys.path.insert(0, str(_CODE_DIR))
    rng = np.random.default_rng(0)
    smiles = [f"C{i % 37}" for i in range(10000)]
    for i in range(200):
        smiles[i] = smiles[0]  # duplicates must stay together
    split = __import__("run_structural_relation_composer_v1").build_split_v1(smiles)
    assert set(split["fit"]) | set(split["monitor"]) | set(split["locked"]) == set(range(10000))
    assert not (set(split["fit"]) & set(split["monitor"]))
    assert not (set(split["fit"]) & set(split["locked"]))
    assert not (set(split["monitor"]) & set(split["locked"]))
    again = __import__("run_structural_relation_composer_v1").build_split_v1(smiles)
    assert np.array_equal(split["fit"], again["fit"])
    dup_rows = {i for i in range(200)}
    where = [("fit", dup_rows <= set(split["fit"])), ("monitor", dup_rows <= set(split["monitor"])), ("locked", dup_rows <= set(split["locked"]))]
    assert any(inside for _, inside in where)


# ---------------------------------------------------------------------------
# slow: real data objects
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_molecule_objects_finite_and_accounted():
    import cscl_features as cf

    ds = cf.load_official_train()
    for i in (0, 1, 2, 5, 9):
        d = ds[i]
        obj = sro.build_mol_objects(i, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        assert np.isfinite(obj.atom_feat).all() and np.isfinite(obj.junc_feat).all() and np.isfinite(obj.global_feat).all()
        assert obj.bond_index.shape[0] == d.edge_index.shape[1] // 2
        assert obj.atom_feat.shape[0] == d.x.shape[0]
        counts = np.zeros(obj.bond_index.shape[0], dtype=np.int64)
        for b1, b2 in obj.junc_bond:
            counts[b1] += 1
            counts[b2] += 1
        assert counts.sum() == 2 * obj.junc_index.shape[0]


@pytest.mark.slow
def test_cached_objects_match_fresh_build():
    cache = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1/bjoc_objects_v1.pt"
    if not cache.exists():
        pytest.skip("object cache not built")
    import cscl_features as cf

    objs = torch.load(cache, map_location="cpu", weights_only=False)
    ds = cf.load_official_train()
    for i in (3, 17, 42):
        fresh = sro.build_mol_objects(i, ds[i].x.numpy(), ds[i].edge_index.numpy(), ds[i].edge_attr.numpy())
        assert np.allclose(fresh.atom_feat, objs[i].atom_feat)
        assert np.array_equal(fresh.bond_index, objs[i].bond_index)
        assert np.array_equal(fresh.junc_bond, objs[i].junc_bond)


@pytest.mark.slow
def test_split_json_matches_rebuild():
    split_json = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1/split_v1.json"
    if not split_json.exists():
        pytest.skip("split not built")
    import cscl_features as cf
    import run_structural_relation_composer_v1 as drv

    payload = json.loads(split_json.read_text())
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    split = drv.build_split_v1(smiles)
    assert payload["n_fit"] == len(split["fit"])
    assert payload["n_monitor"] == len(split["monitor"])
    assert payload["n_locked"] == len(split["locked"])
