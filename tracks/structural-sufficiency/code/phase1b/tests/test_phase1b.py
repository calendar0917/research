"""Fast CPU tests for the Phase 1b structural-relation arms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

from ssrc_data import MolGraph, build_cache, gather_batch, load_zinc_train, zinc_internal_split  # noqa: E402
from ssrc_models import ModelConfig, build_model, parameter_count  # noqa: E402
from ssrc_train import _model_batch  # noqa: E402


def mol_chain(n: int, d: int = 4, cycle: bool = False) -> MolGraph:
    af = np.zeros((n, d), dtype=np.float32)
    af[np.arange(n), np.arange(n) % d] = 1.0
    if cycle:
        bu = np.arange(n, dtype=np.int64)
        bv = (np.arange(n) + 1) % n
    else:
        bu = np.arange(n - 1, dtype=np.int64)
        bv = np.arange(1, n, dtype=np.int64)
    bf = np.zeros((len(bu), 2), dtype=np.float32)
    bf[:, 0] = 1.0
    return MolGraph(af, bu, bv, bf)


def test_zinc_split_matches_tccd_v0() -> None:
    from tracks.ksvd.code.run_tccd_v0 import internal_split

    tr, dev = internal_split(10000)
    s = zinc_internal_split()
    assert np.array_equal(np.asarray(s["fit"]), tr)
    assert np.array_equal(np.asarray(s["monitor"] + s["reserved"]), dev)


def test_models_run_and_share_trunk() -> None:
    cfg = ModelConfig()
    cache = build_cache([mol_chain(6), mol_chain(5, cycle=True)])
    b = gather_batch(cache, [0, 1], torch.tensor([0.0, 1.0]))
    trunks = []
    for name in ["S0", "R1", "M1", "M2"]:
        torch.manual_seed(0)
        model = build_model(name, 4, 2, cfg)
        out = model(_model_batch(name, b))
        assert out.shape == (2,)
        assert torch.isfinite(out).all()
        trunks.append({k: v.clone() for k, v in model.trunk.state_dict().items()})
    for other in trunks[1:]:
        assert all(torch.equal(trunks[0][k], other[k]) for k in trunks[0])


def test_m1_m2_parameter_identical() -> None:
    cfg = ModelConfig()
    torch.manual_seed(0)
    m1 = build_model("M1", 4, 2, cfg)
    torch.manual_seed(0)
    m2 = build_model("M2", 4, 2, cfg)
    assert parameter_count(m1) == parameter_count(m2)
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))


def test_endpoint_swap_invariance() -> None:
    cfg = ModelConfig()
    cache = build_cache([mol_chain(7, cycle=True)])
    b = gather_batch(cache, [0], torch.tensor([0.0]))
    torch.manual_seed(0)
    model = build_model("R1", 4, 2, cfg)
    model.eval()
    b = _model_batch("R1", b)
    with torch.no_grad():
        token = model.trunk(b)
        r = model.rel(token, b["root_edge_index"], b["root_edge_attr"])
        src, dst = b["root_edge_index"]
        lookup = {(int(src[k]), int(dst[k])): k for k in range(src.numel())}
        for (u, v), k in lookup.items():
            if (v, u) in lookup:
                assert torch.allclose(r[k], r[lookup[(v, u)]], atol=1e-6)


def test_s0_ignores_root_edges_and_guards() -> None:
    cfg = ModelConfig()
    cache = build_cache([mol_chain(6)])
    b = gather_batch(cache, [0], torch.tensor([0.0]))
    torch.manual_seed(0)
    model = build_model("S0", 4, 2, cfg)
    model.eval()
    b0 = _model_batch("S0", b)
    bs = dict(b0)
    bs["root_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
    bs["root_edge_attr"] = torch.zeros((0, 2))
    with torch.no_grad():
        assert torch.equal(model(b0), model(bs))
    from ssrc_models import assert_patch_only_batch

    try:
        assert_patch_only_batch(b)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_s0_matches_phase1a_modelS_on_real_lipo() -> None:
    from phase1a_models import ModelS
    from ssrc_data import LIPO_RAW, load_lipo_mols

    mols = load_lipo_mols(LIPO_RAW)[:4]
    cache = build_cache(mols)
    b = _model_batch("S0", gather_batch(cache, [0, 1, 2, 3], torch.arange(4.0)))
    b["patch_mask"] = torch.ones((4, int(cache["mol_n_atoms"][:4].max())), dtype=torch.bool)
    torch.manual_seed(0)
    ref = ModelS(hidden=64, local_layers=2, ffn_layers=2, ffn_hidden=128, dropout=0.1)
    torch.manual_seed(0)
    ours = build_model("S0", 37, 12, ModelConfig())
    mapping = {}
    for k in ours.state_dict():
        if k.startswith("trunk.local.input_proj"):
            nk = "local.input_proj" + k[len("trunk.local.input_proj"):]
        elif k.startswith("trunk.local.layers."):
            parts = k.split(".")
            i, rest = parts[3], ".".join(parts[4:])
            nk = (f"local.convs.{i}." + rest[5:]) if rest.startswith("conv.") else (f"local.norms.{i}." + rest[5:])
        elif k.startswith("trunk.blocks."):
            nk = "blocks." + k[len("trunk.blocks."):]
        else:
            nk = k
        mapping[k] = nk
    ref.load_state_dict({mapping[k]: v for k, v in ours.state_dict().items()})
    ref.eval()
    ours.eval()
    with torch.no_grad():
        assert torch.equal(ref(b), ours(b))


def test_rewire_control() -> None:
    """B1 control: degree-preserving, type-multiset-preserving root rewiring."""
    import numpy as np
    from ssrc_rewire import rewire_mol_bonds, rewire_root_adjacency

    u = np.array([0, 0, 1, 2, 3, 3], dtype=np.int64)
    v = np.array([1, 2, 2, 3, 4, 5], dtype=np.int64)
    attr = np.stack([np.full(6, k, dtype=np.int64) for k in range(4)], axis=1).astype(np.float32)
    u1, v1, attempted, applied = rewire_mol_bonds(u, v, attr, seed=0, mol_id=3)
    assert attempted > 0 and applied > 0
    # no self loops
    assert not np.any(u1 == v1)
    # same edge count, same degree sequence, no duplicate edges
    assert len(u1) == len(u)
    deg = lambda a, b: np.bincount(np.concatenate([a, b]), minlength=6)
    assert np.array_equal(deg(u, v), deg(u1, v1))
    pairs = {(int(a), int(b)) for a, b in zip(u1, v1)}
    assert len(pairs) == len(u1)

    mols, _, _ = load_zinc_train()
    cache = build_cache(mols[:6])
    out, report = rewire_root_adjacency(cache, seed=0)
    n0 = cache["root_edge_start"].diff().numpy()
    n1 = out["root_edge_start"].diff().numpy()
    assert np.array_equal(n0, n1)
    assert report["degree_sequence_preserved"] and report["edge_count_preserved"]
    assert report["total_swaps_applied"] > 0 and report["molecules_changed"] > 0
    # patches untouched
    assert torch.equal(cache["patch_node_feat_flat"], out["patch_node_feat_flat"])
    # root edge count unchanged
    assert int(out["root_edge_src_flat"].shape[0]) == int(cache["root_edge_src_flat"].shape[0])
