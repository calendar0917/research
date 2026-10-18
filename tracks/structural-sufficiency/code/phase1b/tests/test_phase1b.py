"""Fast CPU tests for the Phase 1b structural-relation arms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

from ssrc_data import MolGraph, build_cache, gather_batch, zinc_internal_split  # noqa: E402
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
