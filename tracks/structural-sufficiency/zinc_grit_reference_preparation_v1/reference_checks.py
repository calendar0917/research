"""Functional checks for the frozen GRIT reference.

These checks are deliberately small.  They verify that the RRWP pair/node
inputs exist and are used, that one forward/backward is finite, that the model
has the paper's parameter count, and that eval-mode predictions are invariant
to node renumbering up to float32 noise.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch_geometric.data import Batch, Data


def parameter_check(model: torch.nn.Module, expected: int = 473473) -> dict[str, Any]:
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "actual": actual,
        "paper_zinc": int(expected),
        "difference": actual - int(expected),
        "pass": actual == int(expected),
    }


def dense_rrwp(data: Data, ksteps: int = 21) -> torch.Tensor:
    n = int(data.num_nodes)
    adj = torch.zeros((n, n), dtype=torch.float32, device=data.edge_index.device)
    adj.index_put_((data.edge_index[0], data.edge_index[1]),
                   torch.ones(data.edge_index.size(1),
                              dtype=torch.float32, device=data.edge_index.device),
                   accumulate=True)
    deg = adj.sum(dim=1)
    p = adj / deg.clamp_min(1.0).view(-1, 1)
    p[deg == 0] = 0.0
    pe = [torch.eye(n, dtype=torch.float32, device=adj.device), p]
    for _ in range(2, ksteps):
        pe.append(pe[-1] @ p)
    return torch.stack(pe, dim=-1)


def rrwp_check(data: Data, ksteps: int = 21, atol: float = 1e-6) -> dict[str, Any]:
    if not all(hasattr(data, name) for name in ("rrwp", "rrwp_index", "rrwp_val")):
        return {"pass": False, "reason": "missing rrwp/rrwp_index/rrwp_val"}
    n = int(data.num_nodes)
    expected = dense_rrwp(data, ksteps=ksteps)
    got_pair = data.rrwp_val.reshape(n, n, ksteps)
    pair_abs = float((got_pair - expected).abs().max())
    got_abs = data.rrwp
    expected_diag = expected.diagonal(dim1=0, dim2=1).transpose(0, 1)
    abs_abs = float((got_abs - expected_diag).abs().max())
    return {
        "pass": bool(pair_abs <= atol and abs_abs <= atol),
        "tolerance": atol,
        "node_abs_shape": list(got_abs.shape),
        "pair_index_shape": list(data.rrwp_index.shape),
        "pair_value_shape": list(data.rrwp_val.shape),
        "pair_vs_dense_max_abs": pair_abs,
        "node_abs_vs_dense_max_abs": abs_abs,
        "ksteps": int(ksteps),
    }


def _pred(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    device = next(model.parameters()).device
    batch = batch.to(device)
    model.eval()
    with torch.no_grad():
        pred, _ = model(batch)
    return pred.detach().float().cpu()


def participation_check(model: torch.nn.Module, data: Data) -> dict[str, Any]:
    # GRIT overwrites batch.x in-place with embeddings during forward, so every
    # probe must start from a fresh Batch.
    base = _pred(model, Batch.from_data_list([data.clone()]))
    z = Batch.from_data_list([data.clone()])
    z.rrwp_val = torch.zeros_like(z.rrwp_val)
    pair_delta = float((base - _pred(model, z)).abs().max())
    z = Batch.from_data_list([data.clone()])
    z.rrwp = torch.zeros_like(z.rrwp)
    node_delta = float((base - _pred(model, z)).abs().max())
    return {
        "pass": bool(pair_delta > 1e-7 and node_delta > 1e-7),
        "pair_zero_max_abs_delta": pair_delta,
        "node_abs_zero_max_abs_delta": node_delta,
        "note": "zeroing the RRWP pair/node input must change the prediction",
    }


def _fork_rng(model: torch.nn.Module):
    devices = []
    if torch.cuda.is_available():
        try:
            devices = [torch.cuda.current_device()]
        except Exception:
            devices = []
    return torch.random.fork_rng(devices=devices, enabled=True)


def forward_backward_check(model: torch.nn.Module, data: Data) -> dict[str, Any]:
    probe = copy.deepcopy(model)
    batch = Batch.from_data_list([data.clone()])
    device = next(probe.parameters()).device
    batch = batch.to(device)
    with _fork_rng(probe):
        probe.train()
        probe.zero_grad(set_to_none=True)
        pred, true = probe(batch)
        loss = torch.nn.functional.l1_loss(pred, true.view_as(pred))
        loss.backward()
        grads = [p.grad for p in probe.parameters() if p.grad is not None]
        finite_grads = sum(int(torch.isfinite(g).all()) for g in grads)
        nonzero_grads = sum(int(g.abs().sum() > 0) for g in grads)
    result = {
        "pass": bool(torch.isfinite(loss) and finite_grads == len(grads)
                     and nonzero_grads > 0 and len(grads) > 0),
        "loss": float(loss.detach().cpu()),
        "pred_finite": bool(torch.isfinite(pred).all()),
        "grad_tensors": len(grads),
        "grad_tensors_finite": finite_grads,
        "grad_tensors_nonzero": nonzero_grads,
    }
    del probe, batch, pred, true, loss, grads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def renumbering_check(model: torch.nn.Module, dataset, count: int = 5,
                      atol: float = 5e-6) -> dict[str, Any]:
    from grit.transform.rrwp import add_full_rrwp

    errors = []
    with _fork_rng(model):
        for i in range(count):
            d = dataset[i]
            base = _pred(model, Batch.from_data_list([d.clone()]))
            g = torch.Generator().manual_seed(1000 + i)
            perm = torch.randperm(d.num_nodes, generator=g)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm))
            d2 = Data(
                x=d.x[perm].clone(),
                edge_index=inv[d.edge_index].clone(),
                edge_attr=d.edge_attr.clone(),
                y=d.y.clone(),
            )
            d2 = add_full_rrwp(d2, walk_length=21, add_identity=True)
            perm_pred = _pred(model, Batch.from_data_list([d2]))
            errors.append(float((base - perm_pred).abs().max()))
    return {
        "pass": bool(max(errors, default=float("inf")) <= atol),
        "tolerance": atol,
        "graphs": int(count),
        "max_abs_error": max(errors) if errors else None,
        "per_graph_abs_error": errors,
    }
