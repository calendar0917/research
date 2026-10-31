"""Focused CPU tests for the composition-discovery implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import dc_common as C
import dc_data as D
import dc_models as M


def _views():
    return D.load_views(include_test=False)


def test_distance_bucket_semantics():
    views = _views()
    batch = D.collate(views["valid_cache"], list(range(6)))
    atom_batch = batch["atom_batch"]
    bond_index = batch["bond_index"]
    n_atoms = int(atom_batch.numel())
    adj = [[] for _ in range(n_atoms)]
    for u, v in bond_index.tolist():
        adj[u].append(v)
        adj[v].append(u)

    def bfs(s):
        dist = [-1] * n_atoms
        dist[s] = 0
        queue = [s]
        for u in queue:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    for i in range(n_atoms):
        dist = bfs(i)
        exp34 = sorted(j for j, d in enumerate(dist) if 3 <= d <= 4)
        exp5 = sorted(j for j, d in enumerate(dist) if d >= 5)
        got34 = sorted(batch["d34_idx"][batch["d34_ptr"][i] : batch["d34_ptr"][i + 1]].tolist())
        got5 = sorted(batch["d5_idx"][batch["d5_ptr"][i] : batch["d5_ptr"][i + 1]].tolist())
        assert exp34 == got34
        assert exp5 == got5


def test_no_correction_recovers_base():
    views = _views()
    batch = M.to_device(D.collate(views["valid_cache"], list(range(8))), "cpu")
    base, _ = M.build_family("base", views["spec"], views["train_norm"], 0, "cpu")
    base.eval()
    with torch.no_grad():
        y_base = base(batch)
    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, views["spec"], views["train_norm"], 0, "cpu")
        model.eval()
        out = model.context.w_out if family == "g2l_scalar" else (
            model.context.W_out if family.startswith("g2l") else (model.dist.W_out if family.startswith("dist") else model.pathbind.W_out)
        )
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
            y = model(batch)
        assert float((y - y_base).abs().max()) <= 1e-5


def test_pair_initialisation_parity():
    views = _views()
    for a, b in (("dist_true", "dist_blind"), ("path_bind", "path_unbound")):
        ma, _ = M.build_family(a, views["spec"], views["train_norm"], 0, "cpu")
        mb, _ = M.build_family(b, views["spec"], views["train_norm"], 0, "cpu")
        sa, sb = ma.state_dict(), mb.state_dict()
        assert M.parameter_count(ma) == M.parameter_count(mb)
        assert set(sa) == set(sb)
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_path_binding_invariants():
    views = _views()
    model, _ = M.build_family("path_bind", views["spec"], views["train_norm"], 0, "cpu")
    model.eval()
    batch = M.to_device(D.collate(views["valid_cache"], list(range(16))), "cpu")
    a, _b, h = model._encode(batch)
    with torch.no_grad():
        _, ab = model.pathbind(h, a, batch, "bind")
        _, au = model.pathbind(h, a, batch, "unbound")
    counts = ab["path_counts"]
    gap = (ab["bind"] - au["bind"]).abs().max(dim=1).values
    assert float(gap[counts == 1].max()) <= 1e-6
    assert float(gap[counts >= 2].max()) > 1e-6


def test_renumbering_and_path_order():
    views = _views()
    model, _ = M.build_family("path_bind", views["spec"], views["train_norm"], 0, "cpu")
    model.eval()
    batch = M.to_device(D.collate(views["valid_cache"], list(range(8))), "cpu")
    g = torch.Generator().manual_seed(7)
    permuted = M.to_device(D.permute_batch_atoms(batch, g), "cpu")
    g2 = torch.Generator().manual_seed(8)
    path_perm = M.to_device(D.permute_paths_within_roots(batch, g2), "cpu")
    with torch.no_grad():
        y = model(batch)
        yp = model(permuted)
        ypp = model(path_perm)
    assert float((y - yp).abs().max()) <= 1e-5
    assert float((y - ypp).abs().max()) <= 1e-5
