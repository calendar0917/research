"""Post-hoc binding-gap diagnostic for PATH-BIND / PATH-UNBOUND.

The correction's per-root summary is either ``mean_p(v_p * r_p)`` (BIND) or
``mean_p(v_p) * mean_p(r_p)`` (UNBOUND).  This script measures the norm of the
difference between the two summaries on the frozen validation molecules.  It
is analysis-only and does not change the frozen training protocol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import dc_common as C  # noqa: E402
import dc_data as D  # noqa: E402
import dc_models as M  # noqa: E402


def binding_gap(model, cache, device="cpu", batch_size=64):
    model.eval()
    gaps, bound_norms, rel = [], [], []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = M.to_device(D.collate(cache, rows), device)
            _y, aux = model(batch, return_aux=True)
            v, r = aux["v"], aux["r"]
            path_atom = batch["path_atom"]
            n_roots = int(batch["atom_batch"].shape[0])
            if path_atom.numel() == 0:
                continue
            counts = torch.zeros(n_roots, dtype=v.dtype, device=v.device).index_add_(
                0, path_atom, torch.ones_like(path_atom, dtype=v.dtype)
            )
            has = counts > 0
            sum_vr = torch.zeros((n_roots, v.shape[1]), dtype=v.dtype, device=v.device).index_add_(0, path_atom, v * r)
            sum_v = torch.zeros_like(sum_vr).index_add_(0, path_atom, v)
            sum_r = torch.zeros_like(sum_vr).index_add_(0, path_atom, r)
            bound = sum_vr / counts.clamp(min=1).unsqueeze(1)
            marginal = (sum_v / counts.clamp(min=1).unsqueeze(1)) * (sum_r / counts.clamp(min=1).unsqueeze(1))
            gap = (bound - marginal)[has]
            gaps.append(gap.norm(dim=1).cpu().numpy())
            bound_norms.append(bound[has].norm(dim=1).cpu().numpy())
            rel.append((gap.norm(dim=1) / bound[has].norm(dim=1).clamp(min=1e-12)).cpu().numpy())
    gap = np.concatenate(gaps) if gaps else np.zeros(0)
    bn = np.concatenate(bound_norms) if bound_norms else np.zeros(0)
    rel = np.concatenate(rel) if rel else np.zeros(0)
    return {
        "n_roots_with_paths": int(gap.size),
        "gap_norm_mean": float(gap.mean()) if gap.size else 0.0,
        "gap_norm_median": float(np.median(gap)) if gap.size else 0.0,
        "gap_norm_p95": float(np.quantile(gap, 0.95)) if gap.size else 0.0,
        "gap_norm_max": float(gap.max()) if gap.size else 0.0,
        "bound_norm_mean": float(bn.mean()) if bn.size else 0.0,
        "relative_gap_mean": float(rel.mean()) if rel.size else 0.0,
        "relative_gap_p95": float(np.quantile(rel, 0.95)) if rel.size else 0.0,
    }


def main():
    views = D.load_views(include_test=False, with_distance=False)
    out = {"protocol_id": C.PROTOCOL_ID, "split": "valid420", "families": {}}
    for family in ("path_bind", "path_unbound"):
        arm = C.ARM_SPECS[f"{family}_s0"]
        ckpt = torch.load(C.RUNS / f"{family}_s0" / "checkpoint.pt", map_location="cpu", weights_only=False)
        model, _ = M.build_family(family, views["spec"], views["train_norm"], 0, "cpu")
        model.load_state_dict(ckpt["state_dict"])
        out["families"][family] = binding_gap(model, views["valid_cache"], "cpu")
    path = C.ANALYSIS / "posthoc_path_binding.json"
    C.write_json(path, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
