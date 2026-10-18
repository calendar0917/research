from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPO = CODE.parents[3]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(REPO / "tracks/structural-sufficiency/results/phase0/code"))

import ssrc_toy as toy  # noqa: E402

from phase1a_data import (  # noqa: E402
    LabelVault,
    PatchCache,
    RawMol,
    collate_patch_lists,
    extract_patches,
    load_frozen_split,
    load_raw_graphs,
    patch_multiset_key,
    scaffold_groups_from_split,
)
from phase1a_models import (  # noqa: E402
    GRAPH_ONLY_KEYS,
    PATCH_ONLY_KEYS,
    assert_graph_only_batch,
    assert_patch_only_batch,
    parameter_count,
)
from phase1a_train import TrainConfig, _make_batch, build_models  # noqa: E402

SPLIT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
RAW = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
CSV = REPO / "data/lipophilicity/Lipophilicity.csv"


@pytest.fixture(scope="module")
def split():
    return load_frozen_split(SPLIT)


@pytest.fixture(scope="module")
def raws():
    return load_raw_graphs(RAW)


def test_split_sizes_hashes_and_groups(split) -> None:
    assert len(split.fit) == 2352
    assert len(split.monitor) == 504
    assert len(split.reserved) == 504
    assert len(split.small) == 253
    assert set(split.small) <= set(split.fit)
    assert not (set(split.fit) & set(split.monitor))
    assert not (set(split.fit) & set(split.reserved))
    assert not (set(split.monitor) & set(split.reserved))
    groups = scaffold_groups_from_split(split)
    assert not ({groups[r] for r in split.fit} & {groups[r] for r in split.monitor})
    assert not ({groups[r] for r in split.fit} & {groups[r] for r in split.reserved})
    assert not ({groups[r] for r in split.monitor} & {groups[r] for r in split.reserved})


def test_label_vault_only_retains_allowed(split) -> None:
    vault = LabelVault(CSV, split.small, split.monitor, split.reserved)
    vault.load()
    audit = vault.audit()
    assert audit["reserved_labels_retained"] == 0
    assert audit["labels_retained"] == len(split.small) + len(split.monitor)
    with pytest.raises(PermissionError):
        vault.labels(split.reserved[:1], "train")
    with pytest.raises(PermissionError):
        vault.labels(split.small[:1], "monitor")


def _raw_from_toy(g: toy.AttributedGraph) -> RawMol:
    n = g.n
    ei: list[tuple[int, int]] = []
    ec: list[int] = []
    for u, v, t in g.edges:
        ei.extend([(u, v), (v, u)])
        ec.extend([int(t), int(t)])
    return RawMol(
        mol_index=0,
        smiles="toy",
        atom_class=np.asarray(g.node_types, dtype=np.int64),
        atom_charge=np.zeros(n, dtype=np.int64),
        atom_h=np.zeros(n, dtype=np.int64),
        atom_chirality=np.zeros(n, dtype=np.int64),
        edge_index=np.asarray(ei, dtype=np.int64).T,
        edge_class=np.asarray(ec, dtype=np.int64),
        edge_stereo=np.zeros(len(ec), dtype=np.int64),
        n_components=1,
    )


def test_e2_collision_models_identical() -> None:
    cfg = TrainConfig()
    models = build_models(0, cfg, torch.device("cpu"))
    for ga, gb in (
        (toy.cycle_graph(24, 6), toy.cycle_graph(24, 9)),
        (toy.caterpillar_graph(40, 6), toy.caterpillar_graph(40, 9)),
        (toy.grid_graph(20, 9, 6), toy.grid_graph(20, 9, 9)),
    ):
        pa, pb = extract_patches(_raw_from_toy(ga)), extract_patches(_raw_from_toy(gb))
        assert patch_multiset_key(pa) == patch_multiset_key(pb)
        batch = collate_patch_lists([pa, pb], torch.zeros(2), torch.tensor([0, 1]))
        for name in ("S", "X"):
            models[name].eval()
            with torch.no_grad():
                out = models[name](batch)
            assert torch.allclose(out[0], out[1], atol=1e-6, rtol=0)


def test_patch_order_and_relabel_invariance(raws) -> None:
    cfg = TrainConfig()
    models = build_models(0, cfg, torch.device("cpu"))
    raw = raws[0]
    patches = extract_patches(raw)
    y = torch.zeros(1)
    mi = torch.tensor([0])
    for name in ("S", "X"):
        models[name].eval()
        with torch.no_grad():
            base = models[name](collate_patch_lists([patches], y, mi)).item()
        shuffled = list(patches)
        np.random.default_rng(0).shuffle(shuffled)
        with torch.no_grad():
            v = models[name](collate_patch_lists([shuffled], y, mi)).item()
        assert abs(v - base) <= 1e-6


def test_parameter_match() -> None:
    cfg = TrainConfig()
    models = build_models(0, cfg, torch.device("cpu"))
    s, x = parameter_count(models["S"]), parameter_count(models["X"])
    assert abs(s / x - 1.0) <= 0.05


def test_batch_keys_and_isolation(raws, split) -> None:
    rows = split.small[:4]
    pc = PatchCache.build(raws, rows)
    y = torch.zeros(4)
    pb = _make_batch("S", rows, y, pc)
    gb = _make_batch("G", rows, y, pc)
    assert set(pb) == set(PATCH_ONLY_KEYS)
    assert set(gb) == set(GRAPH_ONLY_KEYS)
    assert_patch_only_batch(pb)
    assert_graph_only_batch(gb)
    assert "edge_index" not in pb
    assert "patch_edge_index" not in gb
    poisoned = dict(pb)
    poisoned["full_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
    with pytest.raises(AssertionError):
        assert_patch_only_batch(poisoned)


def test_s_x_shared_module_init() -> None:
    cfg = TrainConfig()
    models = build_models(3, cfg, torch.device("cpu"))
    for key, value in models["S"].local.state_dict().items():
        assert torch.equal(value, models["X"].local.state_dict()[key])
    for key, value in models["S"].head.state_dict().items():
        assert torch.equal(value, models["X"].head.state_dict()[key])
