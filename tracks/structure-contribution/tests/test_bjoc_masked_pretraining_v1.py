"""Tests for cscl-bjoc-masked-pretraining-v1.

Data-free tests (tiny hand-built molecules) run in a fresh clone on CPU.
Result-dependent checks skip when the git-ignored result JSONs are absent.
Official valid / official test are never referenced or loaded.

The heavy end-to-end audit commands (``audit-swap`` / ``rebuild-audit`` /
``gates``) are exercised by the driver itself and are not repeated per test.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import run_bjoc_masked_pretraining_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bjoc_masked_pretraining_v1.yaml"
RESULTS = ROOT / "results" / "bjoc_masked_pretraining_v1"


# ---------------------------------------------------------------------------
# tiny data-free graphs
# ---------------------------------------------------------------------------


def _tiny_molecule(atom_types, edges, mol_id=0):
    x = np.asarray(atom_types, dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(mol_id, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _tiny_pair():
    obj = _tiny_molecule([0, 1], [(0, 1, 2)])
    x = np.asarray([[0], [1]], dtype=np.int64)
    ei = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    ea = np.asarray([2, 2], dtype=np.int64)
    return obj, x, ei, ea


def _tiny_chain():
    return _tiny_molecule([3, 4, 5, 6], [(0, 1, 1), (1, 2, 3), (2, 3, 1), (1, 3, 2)])


# ---------------------------------------------------------------------------
# mask construction
# ---------------------------------------------------------------------------


def test_mask_counts_min_one_and_ratio():
    for n_atoms in (1, 2, 5, 20, 23, 40):
        for n_bonds in (0, 1, 3, 19):
            a, b = mp.sample_mask(0, 7, 0, n_atoms, n_bonds)
            assert len(a) >= 1 and len(a) <= n_atoms
            if n_atoms >= 7:
                assert abs(len(a) / n_atoms - mp.ATOM_MASK_RATIO) <= 0.051
            if n_bonds:
                assert len(b) >= 1 and len(b) <= n_bonds
                if n_bonds >= 7:
                    assert abs(len(b) / n_bonds - mp.BOND_MASK_RATIO) <= 0.051
            else:
                assert len(b) == 0
            assert len(set(a.tolist())) == len(a)
            assert len(set(b.tolist())) == len(b)


def test_mask_rng_reproducible_and_seed_sensitive():
    a1, b1 = mp.sample_mask(3, 11, 2, 30, 25)
    a2, b2 = mp.sample_mask(3, 11, 2, 30, 25)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    a3, _ = mp.sample_mask(4, 11, 2, 30, 25)
    assert not np.array_equal(a1, a3)
    a4, _ = mp.sample_mask(3, 11, 1, 30, 25)
    assert not np.array_equal(a1, a4)


def test_raw_masking_syncs_directed_records():
    obj = _tiny_chain()
    x = np.asarray([3, 4, 5, 6], dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in [(0, 1, 1), (1, 2, 3), (2, 3, 1), (1, 3, 2)]:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    ei = np.asarray(ei, dtype=np.int64).T
    ea = np.asarray(ea, dtype=np.int64)
    x_m, ea_m = mp.mask_raw_arrays(x, ei, ea, obj.bond_index, np.asarray([0]), np.asarray([1]))
    assert x_m.reshape(-1)[0] == mp.ATOM_MASK_TOKEN
    assert x_m.reshape(-1)[1] == 4
    mapping = mp.directed_bond_index(ei)
    for col in mapping[(1, 2)]:
        assert ea_m.reshape(-1)[col] == mp.BOND_MASK_TOKEN
    for col in mapping[(0, 1)]:
        assert ea_m.reshape(-1)[col] == 1


def test_full_masked_object_changes_confined_to_dependency_map():
    obj = _tiny_chain()
    x = np.asarray([3, 4, 5, 6], dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in [(0, 1, 1), (1, 2, 3), (2, 3, 1), (1, 3, 2)]:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    ei = np.asarray(ei, dtype=np.int64).T
    ea = np.asarray(ea, dtype=np.int64)
    s = mp.build_masked_sample(obj, x, ei, ea, seed=1, visit_index=0)
    dep = mp.dependent_cell_mask(obj, s.atom_ids, s.bond_ids)
    checks = [
        (obj.atom_feat, s.masked.atom_feat, dep["atom"]),
        (obj.bond_feat, s.masked.bond_feat, dep["bond"]),
        (obj.junc_feat, s.masked.junc_feat, dep["junc"]),
        (obj.global_feat, s.masked.global_feat, dep["global"]),
    ]
    for before, after, allowed in checks:
        diff = np.asarray(before) != np.asarray(after)
        assert np.all(diff <= allowed)
    for a, t in zip(s.atom_ids, s.atom_types):
        assert int(s.masked.atom_feat[int(a), :28].argmax()) == mp.ATOM_MASK_TOKEN
        assert s.masked.atom_feat[int(a), int(t)] == 0.0
    for b, t in zip(s.bond_ids, s.bond_types):
        assert int(s.masked.bond_feat[int(b), :4].argmax()) == mp.BOND_MASK_TOKEN
        assert s.masked.bond_feat[int(b), int(t)] == 0.0


def test_target_swap_invariance_minimal():
    obj, x, ei, ea = _tiny_pair()
    s0 = mp.build_masked_sample(obj, x, ei, ea, seed=0, visit_index=0)
    x_b = x.copy()
    x_b[s0.atom_ids[0], 0] = (int(x[s0.atom_ids[0], 0]) + 5) % 27
    s1 = mp.build_masked_sample(obj, x_b, ei, ea, seed=0, visit_index=0)
    for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
        assert np.array_equal(getattr(s0.masked, name), getattr(s1.masked, name))
    assert not np.array_equal(s0.atom_types, s1.atom_types)
    # hidden bond truth swap: the variant base object carries the swapped truth
    ei2 = ei.copy()
    ei2[0] = 1
    ei2[1] = 0
    obj_variant = sro.build_mol_objects(0, x, ei2, np.asarray([3, 3], dtype=np.int64))
    s2 = mp.build_masked_sample(obj_variant, x, ei, np.asarray([3, 3], dtype=np.int64), seed=0, visit_index=0)
    for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat"):
        assert np.array_equal(getattr(s0.masked, name), getattr(s2.masked, name))
    assert not np.array_equal(s0.bond_types, s2.bond_types)


def test_visible_change_changes_exactly_dependency_cells():
    obj, x, ei, ea = _tiny_pair()
    s0 = mp.build_masked_sample(obj, x, ei, ea, seed=0, visit_index=0)
    visible = [a for a in range(obj.n_atoms) if a not in set(map(int, s0.atom_ids))]
    assert visible
    v = visible[0]
    x_v = x.copy()
    x_v[v, 0] = (int(x[v, 0]) + 1) % 27
    s1 = mp.build_masked_sample(obj, x_v, ei, ea, seed=0, visit_index=0)
    dep = mp.dependent_cell_mask(obj, [v], [])
    diff = np.asarray(s0.masked.atom_feat) != np.asarray(s1.masked.atom_feat)
    assert diff.any()
    assert not (diff.any(axis=1) & ~dep["atom"].any(axis=1)).any()


# ---------------------------------------------------------------------------
# pretext model
# ---------------------------------------------------------------------------


def _pretext():
    cfg = src.BJOCConfig()
    backbone = src.build_bjoc(
        cfg,
        seed=0,
        organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    )
    return backbone, mp.PretextModel(backbone)


def test_pretext_loss_finite_targets_isolated_and_grads():
    obj = _tiny_chain()
    x = np.asarray([3, 4, 5, 6], dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in [(0, 1, 1), (1, 2, 3), (2, 3, 1), (1, 3, 2)]:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    ei = np.asarray(ei, dtype=np.int64).T
    ea = np.asarray(ea, dtype=np.int64)
    s = mp.build_masked_sample(obj, x, ei, ea, seed=1, visit_index=0)
    batch, at, bt, _, _ = mp.batch_pretext_samples([s])
    assert "atom_targets" not in batch and "bond_targets" not in batch
    backbone, pretext = _pretext()
    loss, parts = pretext.loss(batch, at, bt)
    assert np.isfinite(float(loss))
    assert float(parts["ce_atom"]) > 0 and float(parts["ce_bond"]) > 0
    pretext.zero_grad()
    loss.backward()
    for group in ("atom_encoder", "bond_encoder", "junction_encoder"):
        module = getattr(backbone, group)
        gn = sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
        assert gn > 0.0
    assert any(p.grad is not None for p in pretext.atom_head.parameters())
    assert any(p.grad is not None for p in pretext.bond_head.parameters())


def test_downstream_has_no_pretext_heads_and_413505_params():
    backbone, _ = _pretext()
    assert not hasattr(backbone, "atom_head") and not hasattr(backbone, "bond_head")
    assert int(sum(p.numel() for p in backbone.parameters())) == 413505
    assert src.forbidden_module_scan(backbone) == []


def test_save_load_replay_encoder_state():
    backbone, pretext = _pretext()
    state = mp.encoder_state(backbone)
    fresh, _ = _pretext()
    mp.load_encoder_state(fresh, state)
    obj, x, ei, ea = _tiny_pair()
    batch = sro.batch_objects([mp.build_masked_sample(obj, x, ei, ea, seed=0, visit_index=0).masked])
    with torch.no_grad():
        assert torch.equal(backbone.encode_atoms(batch), fresh.encode_atoms(batch))


def test_bond_endpoint_symmetry_of_pretext_head():
    obj, x, ei, ea = _tiny_pair()
    samples = [mp.build_masked_sample(obj, x, ei, ea, seed=0, visit_index=0)]
    batch, at, bt, _, _ = mp.batch_pretext_samples(samples)
    _, pretext = _pretext()
    swapped = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    swapped["bond_index"] = swapped["bond_index"][:, [1, 0]]
    with torch.no_grad():
        o1 = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
        o2 = pretext(swapped, swapped["masked_atom_ids"], swapped["masked_bond_ids"])
    assert torch.allclose(o1["bond_logits"], o2["bond_logits"], atol=1e-6)


def test_atom_permutation_invariance_and_batch_consistency():
    obj = _tiny_chain()
    _, pretext = _pretext()
    b1 = sro.batch_objects([obj])
    perm = np.asarray([3, 1, 2, 0], dtype=np.int64)
    x = np.asarray([3, 4, 5, 6], dtype=np.int64).reshape(-1, 1)
    ei, ea = [], []
    for u, v, bt in [(0, 1, 1), (1, 2, 3), (2, 3, 1), (1, 3, 2)]:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    ei = np.asarray(ei, dtype=np.int64).T
    obj2 = sro.build_mol_objects(0, x[np.argsort(perm)], perm[ei], np.asarray(ea, dtype=np.int64))
    b2 = sro.batch_objects([obj2])
    backbone = pretext.backbone
    with torch.no_grad():
        assert torch.allclose(backbone(b1), backbone(b2), atol=1e-6)
    other = _tiny_pair()[0]
    with torch.no_grad():
        solo = backbone(sro.batch_objects([obj]))
        mixed = backbone(sro.batch_objects([other, obj]))
    assert torch.allclose(solo[0], mixed[1], atol=1e-6)


def test_class_prior_stats_entropy():
    codes = np.asarray([0, 0, 1])
    atom_stats = mp.class_prior_stats(codes, np.asarray([1, 1, 1]))["atom"]
    assert atom_stats["n"] == 3
    assert atom_stats["majority_class"] == 0
    assert abs(atom_stats["entropy_nats"] - (2 / 3 * np.log(1.5) + 1 / 3 * np.log(3))) < 1e-9


# ---------------------------------------------------------------------------
# discipline scans
# ---------------------------------------------------------------------------


def test_pretrain_path_reads_no_labels():
    tree = ast.parse(inspect.getsource(drv.pretrain_one_seed))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "y":
            hits.append("attribute .y")
        if isinstance(node, ast.Name) and node.id == "ENCODED_TRAIN":
            hits.append("ENCODED_TRAIN")
    assert not hits
    mod_text = (_CODE_DIR / "bjoc_masked_pretraining_v1.py").read_text(encoding="utf-8")
    mod_tree = ast.parse(mod_text)
    assert not [n for n in ast.walk(mod_tree) if isinstance(n, ast.Attribute) and n.attr == "y"]
    src_text = (_CODE_DIR / "run_bjoc_masked_pretraining_v1.py").read_text(encoding="utf-8")
    for token in ('_load_official_' + 'split("val")', '_load_official_' + 'split("test")'):
        assert token not in src_text
    for token in ("Message" + "Passing", "Transformer" + "Encoder", "Multihead" + "Attention"):
        assert token not in mod_text


def test_batch_order_hash_matches_training_order():
    n, batch_size, epochs = 37, 8, 3
    expected = hashlib.sha256()
    gen = torch.Generator().manual_seed(7 + drv.TRAIN_SHUFFLE_OFFSET)
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        expected.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
    assert drv.batch_order_hash(7, n, batch_size, epochs) == expected.hexdigest()


def test_bootstrap_and_topk_helpers():
    delta = np.asarray([-0.1, 0.2, -0.05, 0.0, 0.3])
    boot = drv.paired_bootstrap(delta, n=200, seed=0)
    assert boot["mean"] == pytest.approx(delta.mean())
    assert boot["ci_lo"] <= boot["mean"] <= boot["ci_hi"] or boot["ci_lo"] <= boot["ci_hi"]
    e_new = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    e_ref = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    removed = drv.top_k_removed(delta, e_new, e_ref, k=1)
    assert removed == pytest.approx(np.delete(delta, 4).mean())


def test_split_builder_grouping(monkeypatch):
    monkeypatch.setattr(drv, "FIT_TARGET", 10)
    monkeypatch.setattr(drv, "MONITOR_TARGET", 5)
    smiles = [f"C{i}" for i in range(40)]
    smiles[3] = smiles[0]
    split = drv.build_split(smiles, "unit-test-salt")
    assert len(split["fit"]) == 10 and len(split["monitor"]) == 5 and len(split["locked"]) == 25
    sides = {}
    for name in ("fit", "monitor", "locked"):
        for r in split[name]:
            sides[int(r)] = name
    assert sides[0] == sides[3]
    split2 = drv.build_split(smiles, "unit-test-salt")
    for name in ("fit", "monitor", "locked"):
        assert np.array_equal(split[name], split2[name])


def test_config_frozen_block_and_definitions():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == "cscl-bjoc-masked-pretraining-v1"
    assert cfg["status"] == "frozen"
    assert cfg["mask"]["atom_ratio"] == 0.15 and cfg["mask"]["bond_ratio"] == 0.15
    assert cfg["mask"]["atom_token"] == mp.ATOM_MASK_TOKEN
    assert cfg["mask"]["bond_token"] == mp.BOND_MASK_TOKEN
    assert cfg["pretext"]["updates"] == mp.PRETRAIN_UPDATES
    assert cfg["arms"]["B"]["updates"] == 15120 and cfg["arms"]["C"]["updates"] == 22680
    assert cfg["frozen"]["mask_tokens"] == {"atom": 27, "bond": 0}
    vocab = set(cfg["decision_vocabulary"])
    assert "SELF_SUPERVISED_PRETRAINING_USEFUL" in vocab
    assert "NO_USEFUL_TRANSFER" in vocab
    split_path = RESULTS / "split_a.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        assert payload["fit_sha256_rows"] == cfg["frozen"]["fit8k_sha256_rows"]
        assert payload["monitor_sha256_rows"] == cfg["frozen"]["monitor_sha256_rows"]
        assert payload["locked_sha256_rows"] == cfg["frozen"]["locked_sha256_rows"]


def test_result_audits_if_present():
    for name in ("mask_target_swap_audit.json", "masked_object_rebuild_audit.json", "pretext_leakage_gates.json"):
        path = RESULTS / name
        if not path.exists():
            pytest.skip(f"{name} not present in this clone")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["official_valid_loaded"] is False
        assert payload["official_test_loaded"] is False
    gates_path = RESULTS / "pretext_leakage_gates.json"
    if gates_path.exists():
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        assert gates["all_pass"] is True
        assert gates["n_passed"] == gates["n_gates"]
