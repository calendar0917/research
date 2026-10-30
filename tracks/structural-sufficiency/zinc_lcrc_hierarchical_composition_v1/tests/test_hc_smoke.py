"""Fast smoke tests for LCRC-HC3-v1 (no GPU, no official test)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import hc_common as C  # noqa: E402
import hc_data as D  # noqa: E402
import hc_models as M  # noqa: E402
import hc_structure as HS  # noqa: E402


def test_enumeration_matches_brute_force():
    bonds = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 2), (0, 2, 0), (3, 4, 0)]
    obj = HS.enumerate_molecule(5, bonds)
    ref = HS.brute_force_directed_paths(5, bonds)
    assert set(map(tuple, obj["path_atom"].tolist())) == \
        set((i, j, k, l) for i, j, k, l, *_ in ref)
    by_key = {tuple(map(int, a)): tuple(map(int, c)) for a, c in
              zip(obj["path_atom"], obj["path_chord"])}
    for i, j, k, l, c_ik, c_jl, c_il in ref:
        assert by_key[(i, j, k, l)] == (c_ik, c_jl, c_il)
    counts = np.zeros(len(bonds), dtype=np.int64)
    for row in range(len(obj["path_atom"])):
        counts[int(obj["path_bonds"][row, 1])] += 1
    assert np.all(counts % 2 == 0)
    assert np.array_equal(counts // 2, np.asarray(obj["bond_npaths"]))


def test_parameter_counts_and_shared_base():
    tr = C.load_cache("train")
    norm = C.fit8k_norm(tr, C.split_rows()["rows"]["fit8k"])
    b, bi = M.build_model("B", norm, 0, "cpu")
    h, hi = M.build_model("H", norm, 0, "cpu")
    f, fi = M.build_model("F", norm, 0, "cpu")
    assert bi["parameters"] == 237_519
    assert hi["parameters"] == 323_391
    assert fi["parameters"] == 323_384
    sb, sh, sf = b.state_dict(), h.state_dict(), f.state_dict()
    assert all(torch.equal(sb[k], sh[k]) for k in sb)
    assert all(torch.equal(sb[k], sf[k]) for k in sb)
    assert torch.equal(h.comp_proj.weight, f.comp_proj.weight)


def test_step0_forward_is_bitwise_identical():
    tr = C.load_cache("train")
    split = C.split_rows()
    norm = C.fit8k_norm(tr, split["rows"]["fit8k"])
    rows = split["rows"]["fit8k"][:8]
    ds = D.SplitData("train", cache=tr)
    batch = ds.batch(rows)
    b, _ = M.build_model("B", norm, 0, "cpu")
    h, _ = M.build_model("H", norm, 0, "cpu")
    f, _ = M.build_model("F", norm, 0, "cpu")
    b.eval(); h.eval(); f.eval()
    with torch.no_grad():
        yb, yh, yf = b(batch), h(batch), f(batch)
    assert torch.equal(yh, yb)
    assert torch.equal(yf, yb)


def test_gradient_path_starts_at_step2():
    tr = C.load_cache("train")
    split = C.split_rows()
    norm = C.fit8k_norm(tr, split["rows"]["fit8k"])
    rows = split["rows"]["fit8k"][:8]
    ds = D.SplitData("train", cache=tr)
    batch = ds.batch(rows)
    target = ds.target_tensor(rows, "cpu")
    for kind in ("H", "F"):
        model, _ = M.build_model(kind, norm, 0, "cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path_grads = []
        for _step in range(2):
            loss = torch.nn.functional.l1_loss(model(batch), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            path_grads.append(float(sum(p.grad.abs().sum() for n, p in model.named_parameters()
                                        if "path_encoder" in n)))
            opt.step()
        assert path_grads[0] == 0.0
        assert path_grads[1] > 0.0
