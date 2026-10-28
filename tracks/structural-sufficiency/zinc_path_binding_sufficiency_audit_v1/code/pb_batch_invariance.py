"""§6 of the brief: the historical LCRC-v1 `N` mean-field is batch-scoped.

`RelationHead.aggregate(mode="N")` computes

    gmean = z.sum(0) / max(z.shape[0], 1)          # z over ALL pairs of the batch
    content[g, r] = gmean   for every non-empty category r of molecule g

so the "mean-field" content of one molecule is the mean over *every* eligible
atom pair in the whole batch (128 molecules during training).  Consequences:

* N's prediction for a molecule depends on which other molecules share its batch;
* the historical G2 batch-invariance gate could not see it because the relation
  head is zero-initialised (so `Δ_rel ≡ 0` at initialisation);
* the R/N contrast therefore compares "true per-molecule correspondence" against
  "loss of correspondence **plus** cross-molecule content leakage", i.e. N is a
  strictly weaker control than the intended one.

This module provides (a) a minimal characterisation with a non-zero relation
head, (b) a corrected per-molecule reference aggregator, and (c) a magnitude
estimate on the frozen trained N checkpoint.  It never modifies LCRC-v1.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

import pb_common as P
import lc_data as LD
import lc_models as M


def per_molecule_N_aggregate(z: torch.Tensor, pair_graph: torch.Tensor,
                             pair_cat: torch.Tensor, n_graphs: int, n_cat: int) -> torch.Tensor:
    """Corrected reference: the mean-field is taken **inside each molecule**."""
    z_dim = z.shape[1]
    content = z.new_zeros((n_graphs, n_cat, z_dim))
    for g in range(int(n_graphs)):
        sel = pair_graph == g
        if not bool(sel.any()):
            continue
        gmean = z[sel].mean(0)
        cats = torch.unique(pair_cat[sel])
        content[g, cats] = gmean.unsqueeze(0)
    return content.view(n_graphs * n_cat, z_dim)


def _delta_for(model, cache, rows, rows_ctx, *, mode, n_cat, corrected=False):
    """Δ_rel of the molecules in ``rows`` inside a batch that also holds ``rows_ctx``."""
    all_rows = list(dict.fromkeys([int(r) for r in rows] + [int(r) for r in rows_ctx]))
    idx = {int(r): i for i, r in enumerate(all_rows)}
    batch = LD.collate(cache, all_rows)
    model.eval()
    with torch.no_grad():
        a, b, h = model._encode(batch)
        z = model.rel.pair_content(h, batch["pair_i"], batch["pair_j"])
        n_graphs = int(batch["n_graphs"])
        if corrected:
            content = per_molecule_N_aggregate(z, batch["pair_graph"], batch["pair_cat"],
                                               n_graphs, n_cat)
            count = z.new_zeros((n_graphs * n_cat,))
            count.index_add_(0, batch["pair_graph"] * n_cat + batch["pair_cat"],
                             torch.ones_like(batch["pair_cat"], dtype=z.dtype))
            content = content.view(n_graphs, n_cat, -1)
            count = count.view(n_graphs, n_cat)
            exists = (count > 0).to(z.dtype)
        else:
            content, count, exists = model.rel.aggregate(
                z, batch["pair_graph"], batch["pair_cat"], n_graphs, mode)
        delta = model.rel.readout(content, count, exists)
    return {int(r): float(delta[idx[int(r)]]) for r in rows}


def _nonzero_head(model, seed: int, scale: float = 0.05) -> None:
    """Give the relation head a fixed non-zero weight (label-free, deterministic)."""
    torch.manual_seed(int(seed))
    with torch.no_grad():
        fan_in = model.rel.head.weight.shape[1]
        model.rel.head.weight.normal_(0.0, scale / np.sqrt(fan_in))
        model.rel.head.bias.normal_(0.0, scale)


def _batch_dependence(model, cache, probe_rows, contexts, *, mode, corrected, seed):
    """max |Δ(m | batch b1) − Δ(m | batch b2)| over context pairs."""
    vals = {}
    for ci, ctx in enumerate(contexts):
        vals[ci] = _delta_for(model, cache, probe_rows, ctx, mode=mode,
                              n_cat=model.rel.n_cat, corrected=corrected)
    diffs = []
    for m in probe_rows:
        v = [vals[ci][int(m)] for ci in range(len(contexts))]
        diffs.append(max(v) - min(v))
    return {"per_molecule_range": [float(d) for d in diffs],
            "max_abs_range": float(max(diffs)) if diffs else 0.0,
            "mean_abs_range": float(np.mean(diffs)) if diffs else 0.0,
            "values_per_context": {str(k): v for k, v in vals.items()}}


def run_batch_invariance_audit(cache: Mapping[str, Any], norm: Mapping[str, Any],
                               *, n_probe: int = 4, seed: int = 20261030) -> dict:
    """Minimal, label-free characterisation of the batch-scoped N mean field."""
    rows = list(range(0, 32))  # molecule ids used as batch context
    probe = rows[:n_probe]
    contexts = [probe, rows[:8], rows[:32]]  # same molecules, growing batch
    torch.manual_seed(seed)
    model_N = M.LCRC("N", norm)
    model_N.add_relation()
    _nonzero_head(model_N, seed)
    model_N.eval()
    model_R = M.LCRC("R", norm)
    model_R.add_relation()
    with torch.no_grad():  # identical head parameters for the R comparison
        model_R.rel.load_state_dict(model_N.rel.state_dict())
    model_R.eval()

    out = {
        "setting": "random-init LCRC bodies, identical non-zero relation head, "
                   "frozen A-free (label-free), probe molecules 0..3",
        "probe_rows": probe,
        "batch_sizes": [len(c) for c in contexts],
        "historical_N": _batch_dependence(model_N, cache, probe, contexts,
                                          mode="N", corrected=False, seed=seed),
        "R_reference": _batch_dependence(model_R, cache, probe, contexts,
                                        mode="R", corrected=False, seed=seed),
        "corrected_N": _batch_dependence(model_N, cache, probe, contexts,
                                        mode="N", corrected=True, seed=seed),
    }

    # magnitude on the *trained* N_s0 checkpoint (read-only; used only to show
    # that the defect survives training, never to reinterpret LCRC-v1 numbers)
    try:
        path = P.L.REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1/runs/lcrcN_s0/best.pt"
        sha = P.sha256_file(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        trained = M.load_model_from_state("N", norm, ck["state_dict"], "cpu", cfg=ck["config"])
        trained.eval()
        big = list(range(0, 128))
        ctx = [probe, big]
        out["trained_N_s0"] = {
            "checkpoint_sha256": sha,
            "frozen_valid_mae": float(ck["valid_mae"]),
            "head_norm": float(torch.linalg.vector_norm(trained.rel.head.weight)),
            "batch_sizes": [len(c) for c in ctx],
            **_batch_dependence(trained, cache, probe, ctx, mode="N",
                                corrected=False, seed=seed),
        }
        out["trained_N_s0"]["corrected"] = _batch_dependence(
            trained, cache, probe, ctx, mode="N", corrected=True, seed=seed)
    except Exception as exc:  # noqa: BLE001
        out["trained_N_s0"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


__all__ = ["per_molecule_N_aggregate", "run_batch_invariance_audit"]
